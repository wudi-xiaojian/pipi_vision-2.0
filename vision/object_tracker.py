"""多目标追踪模块（去残影优化版）。

核心改进：
1. 缩短最大滑行帧数 (max_coast_frames)，快速清除丢失目标的残影。
2. 提高关联 IoU 阈值，防止错误匹配导致的轨迹漂移。
3. 新增 _is_duplicate_spawn 检查，防止在同一位置生成多个新轨迹。
4. 保持高响应卡尔曼参数，确保跟得上移动。
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Sequence

import cv2
import numpy as np

from .object_detector import (
    Detection,
    ObjectDetector,
    ObjectSpec,
)


@dataclass
class TrackedObject:
    object_id: str
    name_cn: str
    track_id: int

    bbox: list[float]
    center: list[float]
    width: float
    height: float
    confidence: float

    kf_state: np.ndarray
    kf_covariance: np.ndarray
    kf: Any

    hits: int = 0
    coasting_frames: int = 0
    is_predicted: bool = False
    confirmed: bool = False
    first_frame: int = 0
    last_seen_frame: int = 0
    trail: list[list[int]] = field(default_factory=list)

    # 真实检测帧之间的测量速度。
    # 注意：这里只用于显示/Activity Engine，不参与任何追踪决策。
    measured_speed_px_s: float = 0.0
    last_measured_center: list[float] | None = None
    last_measured_frame: int | None = None

    # 速度鲁棒化：保存最近几次“真实检测”得到的原始速度。
    # 这些字段只用于速度估计，不参与追踪匹配。
    speed_history: list[float] = field(default_factory=list)
    last_raw_speed_px_s: float = 0.0

    @property
    def speed_px_s(self) -> float:
        return float(self.measured_speed_px_s)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        del d["kf_state"]
        del d["kf_covariance"]
        del d["kf"]
        d["trail"] = self.trail[-30:]
        d["speed_px_s"] = round(self.speed_px_s, 1)
        # 内部速度滤波状态不写入对外 JSON。
        d.pop("speed_history", None)
        d.pop("last_raw_speed_px_s", None)
        if isinstance(self.bbox, np.ndarray):
            d["bbox"] = self.bbox.tolist()
        if isinstance(self.center, np.ndarray):
            d["center"] = self.center.tolist()
        return d


class KalmanBoxTracker:
    def __init__(self, bbox: list[float], dt: float = 1/30):
        x, y, w, h = bbox
        self.state = np.array([x, y, w, h, 0, 0], dtype=np.float64)
        self.P = np.eye(6) * 10

        # Q: 过程噪声 - 保持较高以允许快速移动
        self.Q = np.eye(6)
        self.Q[0, 0] = 1.0
        self.Q[1, 1] = 1.0
        self.Q[2, 2] = 1.0
        self.Q[3, 3] = 1.0
        self.Q[4, 4] = 0.5
        self.Q[5, 5] = 0.5

        # R: 测量噪声 - 保持较低以信任检测器
        self.R = np.eye(4) * 5.0

        self.F = np.eye(6)
        self.F[0, 4] = dt
        self.F[1, 5] = dt

        self.H = np.zeros((4, 6))
        self.H[0, 0] = 1
        self.H[1, 1] = 1
        self.H[2, 2] = 1
        self.H[3, 3] = 1

        self.dt = dt

    def predict(self) -> np.ndarray:
        self.state = np.dot(self.F, self.state)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        return self.state[:4]

    def update(self, z: np.ndarray):
        y_res = z - np.dot(self.H, self.state)

        # 如果偏差过大，直接重置，避免产生长距离的“鬼影”拉伸
        dist = np.linalg.norm(y_res[:2])
        if dist > 100:
             self.state[:4] = z
             self.state[4:] = 0
             self.P = np.eye(6) * 10
             return

        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        try:
            K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        except np.linalg.LinAlgError:
            return

        self.state = self.state + np.dot(K, y_res)
        I = np.eye(6)
        self.P = np.dot((I - np.dot(K, self.H)), self.P)

    def get_bbox_xyxy(self) -> list[float]:
        x, y, w, h = self.state[:4]
        w = max(w, 1)
        h = max(h, 1)
        x1 = x - w / 2
        y1 = y - h / 2
        x2 = x + w / 2
        y2 = y + h / 2
        return [x1, y1, x2, y2]

    def get_center(self) -> list[float]:
        return [float(self.state[0]), float(self.state[1])]


def _iou_box(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


class SimpleTracker:
    def __init__(
        self,
        match_iou: float = 0.25,     # 与 YAML 保持一致
        max_coast_frames: int = 5,   # 【关键】大幅缩短滑行时间，快速清除残影
        min_hits: int = 3,
        merge_iou: float = 0.6,
        merge_contain: float = 0.8,
        merge_streak: int = 3,
        speed_history_size: int = 5,
        speed_ema_alpha: float = 0.35,
        max_center_jump_ratio: float = 3.0,
        low_confidence_threshold: float = 0.10,
        coast_speed_decay: float = 0.80,
        fps: float = 30.0,
    ):
        self.match_iou = match_iou
        self.max_coast_frames = max_coast_frames
        self.min_hits = min_hits
        self.merge_iou = merge_iou
        self.merge_contain = merge_contain
        self.merge_streak = merge_streak
        self.fps = fps
        self.dt = 1.0 / fps

        self._streaks: dict[tuple[int, int], int] = {}
        self.tracks: dict[int, TrackedObject] = {}
        self._next_id = 1

        # ---------------------------------------------------------
        # 速度测量参数
        # ---------------------------------------------------------
        # 使用短窗口中位数抑制检测框抖动造成的单点尖峰，
        # 再通过 EMA 让 Activity Engine 看到的速度更稳定。
        self.speed_history_size = max(1, int(speed_history_size))
        self.speed_ema_alpha = min(1.0, max(0.0, float(speed_ema_alpha)))
        # 单次真实检测之间，允许的最大中心位移比例。
        # 以物体自身尺寸作为尺度，避免使用一个全局像素阈值。
        self.max_center_jump_ratio = max(0.0, float(max_center_jump_ratio))
        # 低置信度检测更容易出现 bbox 抖动，因此对极端跳变更严格。
        self.low_confidence_threshold = min(1.0, max(0.0, float(low_confidence_threshold)))
        # 丢失检测后的速度衰减；不直接使用 Kalman 预测速度覆盖测量速度。
        self.coast_speed_decay = min(1.0, max(0.0, float(coast_speed_decay)))

    def step(
        self, dets: list[Detection], frame_idx: int, fps: float
    ) -> list[TrackedObject]:
        self.dt = 1.0 / (fps if fps > 0 else 30.0)

        # 1) 预测
        predicted_boxes: dict[int, list[float]] = {}
        for t in self.tracks.values():
            pred_cxcywh = t.kf.predict()
            cx, cy, w, h = pred_cxcywh
            predicted_boxes[t.track_id] = [cx - w/2, cy - h/2, cx + w/2, cy + h/2]

        # 2) 关联
        matched, unmatched_tracks, unmatched_dets = self._associate(dets, predicted_boxes)

        # 3) 更新匹配轨迹
        for tid, det in matched:
            t = self.tracks[tid]
            x1, y1, x2, y2 = det.bbox_xyxy
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            w, h = x2 - x1, y2 - y1
            z = np.array([cx, cy, w, h], dtype=np.float64)

            # ---------------------------------------------------------
            # 速度只根据“实际检测到的中心点”计算。
            #
            # 重要：
            # 1. 不修改 Kalman 的预测/更新逻辑。
            # 2. 不把速度用于匹配、ID、coasting、merge 等追踪决策。
            # 3. 只作为 speed_px_s 的显示/Activity Engine 数据。
            # ---------------------------------------------------------
            if (
                t.last_measured_center is not None
                and t.last_measured_frame is not None
            ):
                frame_delta = frame_idx - t.last_measured_frame

                if frame_delta > 0:
                    dx = cx - t.last_measured_center[0]
                    dy = cy - t.last_measured_center[1]
                    distance = math.hypot(dx, dy)
                    dt_seconds = frame_delta / (fps if fps > 0 else 30.0)

                    if dt_seconds > 0:
                        raw_speed = distance / dt_seconds
                        t.last_raw_speed_px_s = raw_speed

                        # -------------------------------------------------
                        # P0：过滤明显异常的中心跳变
                        # -------------------------------------------------
                        # 使用上一次真实检测时的 bbox 尺寸作为尺度。
                        # 正常情况下，同一个物体在一次检测间隔内不应
                        # 跨越数倍自身尺寸；异常跳变通常来自检测框抖动。
                        previous_scale = max(
                            float(t.width),
                            float(t.height),
                            1.0,
                        )
                        max_allowed_distance = (
                            previous_scale
                            * self.max_center_jump_ratio
                        )

                        extreme_jump = (
                            distance > max_allowed_distance
                            and det.confidence < self.low_confidence_threshold
                        )

                        if not extreme_jump:
                            t.speed_history.append(raw_speed)
                            if len(t.speed_history) > self.speed_history_size:
                                t.speed_history.pop(0)

                            # 中位数先去掉单次尖峰，再做 EMA。
                            median_speed = float(
                                np.median(
                                    np.asarray(t.speed_history, dtype=np.float64)
                                )
                            )

                            if t.measured_speed_px_s <= 0.0:
                                t.measured_speed_px_s = median_speed
                            else:
                                t.measured_speed_px_s = (
                                    (1.0 - self.speed_ema_alpha)
                                    * t.measured_speed_px_s
                                    + self.speed_ema_alpha
                                    * median_speed
                                )

            # 无论本次速度是否被过滤，都更新“最近一次真实检测”的位置。
            # 这样异常检测不会让旧位置永久参与后续速度计算。
            t.last_measured_center = [float(cx), float(cy)]
            t.last_measured_frame = frame_idx

            t.kf.update(z)

            best_bbox = t.kf.get_bbox_xyxy()
            t.bbox = best_bbox
            t.center = t.kf.get_center()
            t.width = best_bbox[2] - best_bbox[0]
            t.height = best_bbox[3] - best_bbox[1]
            t.confidence = det.confidence
            t.hits += 1
            t.coasting_frames = 0
            t.is_predicted = False
            t.confirmed = t.hits >= self.min_hits
            t.last_seen_frame = frame_idx
            t.trail.append([int(t.center[0]), int(t.center[1])])
            if len(t.trail) > 60:
                t.trail.pop(0)

        # 4) 未匹配轨迹 -> 滑行 (由于 max_coast_frames 很小，这里很快会进入删除逻辑)
        for tid in unmatched_tracks:
            t = self.tracks[tid]
            pred_bbox = predicted_boxes[tid]
            t.bbox = pred_bbox
            t.center = [(pred_bbox[0] + pred_bbox[2])/2, (pred_bbox[1] + pred_bbox[3])/2]
            t.coasting_frames += 1
            t.is_predicted = True
            t.confidence *= 0.90 # 更快降低置信度

            # 预测/滑行帧不能拿 Kalman 的预测位移重新计算 measured_speed。
            # 只让上一段真实测量速度逐步衰减，避免“丢检后还一直高速”。
            t.measured_speed_px_s *= self.coast_speed_decay

        # 5) 未匹配检测 -> 新建 (增加去重检查)
        for det in unmatched_dets:
            if self._is_supplementary(det):
                continue
            if self._is_duplicate_spawn(det):
                continue
            self._spawn(det, frame_idx)

        # 6) 清理 (更激进的清理)
        dead = [tid for tid, t in self.tracks.items() if t.coasting_frames > self.max_coast_frames]
        for tid in dead:
            del self.tracks[tid]

        # 7) 合并
        self._merge_tracks()

        return sorted(self.tracks.values(), key=lambda t: -t.confidence)

    def _is_duplicate_spawn(self, det: Detection) -> bool:
        """检查是否已经有非常接近的轨迹，防止生成残影"""
        cx, cy = det.center_xy
        for t in self.tracks.values():
            if t.object_id != det.object_id:
                continue
            # 计算中心点距离
            dist = math.hypot(t.center[0] - cx, t.center[1] - cy)
            # 如果距离小于宽高的一半，认为已经是同一个物体，只是没匹配上（可能是IoU阈值太高）
            threshold = max(t.width, t.height) * 0.5
            if dist < threshold:
                return True
        return False

    def _is_supplementary(self, det: Detection) -> bool:
        cx, cy = det.center_xy
        for t in self.tracks.values():
            if t.object_id != det.object_id:
                continue
            x1, y1, x2, y2 = t.bbox
            if x1 - 10 <= cx <= x2 + 10 and y1 - 10 <= cy <= y2 + 10:
                return True
        return False

    def _associate(self, dets, predicted_boxes):
        cands = []
        for tid, t in self.tracks.items():
            for di, d in enumerate(dets):
                if d.object_id != t.object_id:
                    continue
                iou = _iou_box(predicted_boxes[tid], d.bbox_xyxy)
                if iou >= self.match_iou:
                    cands.append((iou, d.confidence, tid, di))

        cands.sort(key=lambda c: (-c[0], -c[1]))
        used_t = set()
        used_d = set()
        matched = []

        for iou, conf, tid, di in cands:
            if tid in used_t or di in used_d:
                continue
            used_t.add(tid)
            used_d.add(di)
            matched.append((tid, dets[di]))

        unmatched_tracks = [tid for tid in self.tracks if tid not in used_t]
        unmatched_dets = [d for i, d in enumerate(dets) if i not in used_d]
        return matched, unmatched_tracks, unmatched_dets

    def _spawn(self, det: Detection, frame_idx: int):
        x1, y1, x2, y2 = det.bbox_xyxy
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        w, h = x2 - x1, y2 - y1

        kf = KalmanBoxTracker([cx, cy, w, h], dt=self.dt)

        t = TrackedObject(
            object_id=det.object_id,
            name_cn=det.name_cn,
            track_id=self._next_id,
            bbox=[x1, y1, x2, y2],
            center=[cx, cy],
            width=w,
            height=h,
            confidence=det.confidence,
            kf_state=kf.state.copy(),
            kf_covariance=kf.P.copy(),
            kf=kf,
            first_frame=frame_idx,
            last_seen_frame=frame_idx,
            hits=1,
            trail=[[int(cx), int(cy)]],
            measured_speed_px_s=0.0,
            last_measured_center=[float(cx), float(cy)],
            last_measured_frame=frame_idx,
            speed_history=[],
            last_raw_speed_px_s=0.0,
        )
        self.tracks[self._next_id] = t
        self._next_id += 1

    def _merge_tracks(self):
        by_obj: dict[str, list[TrackedObject]] = {}
        for t in self.tracks.values():
            by_obj.setdefault(t.object_id, []).append(t)

        victims: set[int] = set()
        current_overlapping_pairs: set[tuple[int, int]] = set()

        for group in by_obj.values():
            group.sort(key=lambda t: -t.confidence)
            for i, a in enumerate(group):
                if a.track_id in victims:
                    continue
                for j, b in enumerate(group):
                    if j <= i:
                        continue
                    if b.track_id in victims:
                        continue

                    iou = _iou_box(a.bbox, b.bbox)
                    ax1, ay1, ax2, ay2 = a.bbox
                    bx1, by1, bx2, by2 = b.bbox
                    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
                    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
                    inter = iw * ih
                    area_a = (ax2 - ax1) * (ay2 - ay1)
                    area_b = (bx2 - bx1) * (by2 - by1)
                    min_area = min(area_a, area_b)
                    cont = inter / min_area if min_area > 0 else 0

                    if iou >= self.merge_iou or cont >= self.merge_contain:
                        pair = (min(a.track_id, b.track_id), max(a.track_id, b.track_id))
                        current_overlapping_pairs.add(pair)
                        streak = self._streaks.get(pair, 0) + 1
                        self._streaks[pair] = streak
                        if streak >= self.merge_streak:
                            victims.add(b.track_id)
                            a.hits += b.hits // 2

        for pair in list(self._streaks.keys()):
            if pair not in current_overlapping_pairs:
                self._streaks[pair] = 0

        for tid in victims:
            self.tracks.pop(tid, None)


class TrackerFrameResult:
    def __init__(self, frame_idx, detections, tracks, infer_ms, ran_detection):
        self.frame_idx = frame_idx
        self.detections = detections
        self.tracks = tracks
        self.infer_ms = infer_ms
        self.ran_detection = ran_detection

    def count_confirmed(self, object_id: str) -> int:
        return sum(1 for t in self.tracks if t.object_id == object_id and t.confirmed)


class ObjectTracker:
    def __init__(
        self,
        specs: Sequence[ObjectSpec],
        *,
        model_size: str = "s",
        device: str = "auto",
        imgsz: int = 640,
        dup_iou: float = 0.65,
        detect_every: int = 1,
        match_iou: float = 0.25,
        max_coast_frames: int = 5,
        min_hits: int = 3,
        merge_iou: float = 0.6,
        merge_contain: float = 0.8,
        merge_streak: int = 3,
        speed_history_size: int = 5,
        speed_ema_alpha: float = 0.35,
        max_center_jump_ratio: float = 3.0,
        low_confidence_threshold: float = 0.10,
        coast_speed_decay: float = 0.80,
        fps: float = 30.0,
        verbose: bool = False,
    ):
        self.detector = ObjectDetector(
            specs, model_size=model_size, device=device, imgsz=imgsz, dup_iou=dup_iou, verbose=verbose
        )
        self.tracker = SimpleTracker(
            match_iou=match_iou,
            max_coast_frames=max_coast_frames,
            min_hits=min_hits,
            merge_iou=merge_iou,
            merge_contain=merge_contain,
            merge_streak=merge_streak,
            speed_history_size=speed_history_size,
            speed_ema_alpha=speed_ema_alpha,
            max_center_jump_ratio=max_center_jump_ratio,
            low_confidence_threshold=low_confidence_threshold,
            coast_speed_decay=coast_speed_decay,
            fps=fps,
        )
        self.detect_every = max(1, detect_every)
        self.fps = fps
        self.frame_idx = 0
        self.infer_times = []
        self._color_of = {sp.id: sp.color for sp in self.detector.specs}

    def process(self, frame: np.ndarray) -> TrackerFrameResult:
        ran = self.frame_idx % self.detect_every == 0
        dets = []
        infer_ms = 0.0
        if ran:
            res = self.detector.detect(frame)
            dets = res.detections
            infer_ms = res.infer_ms
            self.infer_times.append(infer_ms)

        tracks = self.tracker.step(dets, self.frame_idx, self.fps)
        out = TrackerFrameResult(self.frame_idx, dets, tracks, infer_ms, ran)
        self.frame_idx += 1
        return out

    def draw(self, frame: np.ndarray, result: TrackerFrameResult) -> np.ndarray:
        canvas = frame.copy()
        for t in result.tracks:
            color = self._color_of.get(t.object_id, (0, 200, 0))
            x1, y1, x2, y2 = (int(v) for v in t.bbox)

            if t.is_predicted:
                # 预测态画虚线，提示用户这是“幽灵”状态，即将消失
                cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 1, cv2.LINE_4)
            else:
                cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)

            for i in range(1, len(t.trail)):
                cv2.line(canvas, tuple(t.trail[i-1]), tuple(t.trail[i]), color, 1)

            label = f"{t.name_cn}#{t.track_id}"
            if t.is_predicted:
                label += f" (P{t.coasting_frames})"
            cv2.putText(canvas, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        return canvas