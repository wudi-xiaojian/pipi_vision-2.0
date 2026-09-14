"""多目标追踪模块（方案第 8 节 / 实验 3 的前半）。

职责：在检测之上维持"同一个物品跨帧是同一个"的身份，
输出带速度、丢失状态的结构化轨迹，供 Activity Engine 判断
拿起 / 移动 / 放置 / 堆叠。

核心解决两个问题：
1. 身份：这一帧的杯子和上一帧的杯子是不是同一个？（track_id 稳定）
2. 遮挡：手握住杯子导致检测丢失时，用速度预测继续维持 track_id
   （coasting / 惯性滑行），而不是当场断轨。
   遮挡一消失，杯子回到预测位置附近 → 抢回原 ID。

为什么手写关联而不用 ultralytics 内置 ByteTrack：
- 本项目关联有强先验：只在同 object_id 内部关联、需要像素级速度
  给状态机用、遮挡期间要有显式的"预测中"状态供 VLM 触发器消费。
  自写版本行为完全可见、阈值完全可解释。
- 真实素材上若出现频繁 ID 切换，再无痛替换为
  model.track(persist=True) 的 ByteTrack 实现即可，接口不变。

实测发现的坑（demo 视频暴露，对应两道防御）：
  模型对同一个杯子会同时输出"整杯框"和"下半截框"，两框
  IoU≈0.47：低于检测去重阈值(0.65)逃过合并，又高到能被关联，
  于是同一物体养出两条平行轨迹（25 条轨迹挤在 3 个杯子上）。
  防御 A merge_overlapping()：同类两条轨迹 IoU≥merge_iou 或
    小框被大框吞没≥merge_contain 时只留强者。
  防御 B 新生抑制：未匹配检测框中心落在已有轨迹框内时，视为
    同一物体的补充框，不另起新轨迹。

下游用法（对应方案第 9 节"动作 = 时间上的变化"）：
    for track in tracker.tracks():
        track.vx / track.vy        # 像素/秒，EMA 平滑
        track.is_predicted         # True = 这一帧是预测位置，不是观测
        track.coasting_frames      # 已连续丢失多少帧
        track.confirmed            # 出现 >= min_hits 帧（滤掉闪现误检）
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
    """一个被维持身份的物体轨迹（含预测态）。"""

    object_id: str
    name_cn: str
    track_id: int
    bbox: list[float]          # 观测态=观测框；预测态=外推框
    center: list[float]
    width: float
    height: float
    confidence: float          # 预测态会随丢失逐帧衰减
    vx: float = 0.0            # 像素/秒（EMA 平滑，含机位移动）
    vy: float = 0.0
    hits: int = 0              # 被观测确认的总次数
    coasting_frames: int = 0   # 已连续丢失的帧数（0=本帧有观测）
    is_predicted: bool = False
    confirmed: bool = False    # hits >= min_hits（tentative 轨迹为 False）
    first_frame: int = 0
    last_seen_frame: int = 0
    trail: list[list[int]] = field(default_factory=list)  # 最近观测中心点

    @property
    def speed_px_s(self) -> float:
        return math.hypot(self.vx, self.vy)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["trail"] = self.trail[-30:]
        d["speed_px_s"] = round(self.speed_px_s, 1)
        return d


def _iou(a: Sequence[float], b: Sequence[float]) -> float:
    return ObjectDetector._iou(a, b)


class SimpleTracker:
    """恒速预测 + 贪心 IoU 关联的轻量多目标追踪器。

    参数说明：
      match_iou:        预测框与检测框 IoU >= 此值才算同一物体。
      max_coast_frames: 连续丢失超过此帧数则丢弃轨迹。
                        30fps 下 75 ≈ 2.5 秒遮挡容忍。
      min_hits:         观测确认次数达到此值才转 confirmed。
                        单帧闪现的误检永远进不了 confirmed，
                        这是第一道免费的去误检闸门。
      merge_iou:        两条同类轨迹 IoU≥此值视为同一物体，合并。
                        针对"整杯框+半截框"的影子轨迹。
      merge_contain:    小框 >= 此比例被大框覆盖也触发合并。
      merge_streak:     重叠需连续保持多少帧才真正合并。
                        影子框（同一物体的两部分检出）永久重叠；
                        两个真实杯子交叉穿行只重叠 1~2 帧。
                        用持续性区分这两种情况，交叉不串、影子能杀。
      max_speed:        速度上限（像素/秒）。一次错配的残差除以 dt
                        会放大几十倍，钳制防止恒速外推把轨迹带飞。
    """

    def __init__(
        self,
        match_iou: float = 0.25,
        max_coast_frames: int = 75,
        min_hits: int = 3,
        merge_iou: float = 0.55,
        merge_contain: float = 0.75,
        merge_streak: int = 10,
        max_speed: float = 1500.0,
    ) -> None:
        self.match_iou = match_iou
        self.max_coast_frames = max_coast_frames
        self.min_hits = min_hits
        self.merge_iou = merge_iou
        self.merge_contain = merge_contain
        self.merge_streak = merge_streak
        self.max_speed = max_speed
        self._streaks: dict[tuple[int, int], int] = {}
        self.tracks: dict[int, TrackedObject] = {}
        self._next_id = 1

    # ---------- 主循环 ----------

    def step(
        self, dets: list[Detection], frame_idx: int, fps: float
    ) -> list[TrackedObject]:
        dt = 1.0 / (fps if fps > 0 else 30.0)

        # 1) 恒速预测：所有轨迹先外推一帧
        predicted: dict[int, list[float]] = {}
        for t in self.tracks.values():
            dx, dy = t.vx * dt, t.vy * dt
            x1, y1, x2, y2 = t.bbox
            predicted[t.track_id] = [x1 + dx, y1 + dy, x2 + dx, y2 + dy]

        # 2) 关联：同 object_id 内，按 (IoU, 置信度) 贪心配对
        matched, unmatched_tracks, unmatched_dets = self._associate(
            dets, predicted
        )

        # 3) 更新被匹配轨迹（观测修正预测）
        for tid, det in matched:
            t = self.tracks[tid]
            self._update_from_obs(t, det, predicted[tid], frame_idx, dt)

        # 4) 没匹配上的旧轨迹 → 进入滑行（coasting）
        for tid in unmatched_tracks:
            t = self.tracks[tid]
            t.bbox = predicted[tid]
            t.center = [
                (t.bbox[0] + t.bbox[2]) / 2.0,
                (t.bbox[1] + t.bbox[3]) / 2.0,
            ]
            t.coasting_frames += 1
            t.is_predicted = True
            t.confidence = round(t.confidence * 0.92, 4)  # 丢失越久越不可信

        # 5) 没匹配上的新检测 → 生成 tentative 轨迹
        #    （新生抑制：中心落在已有轨迹框内的检测视为补充框，不起新轨）
        for det in unmatched_dets:
            if self._is_supplementary(det):
                continue
            self._spawn(det, frame_idx)

        # 6) 清理丢太久的轨迹
        dead = [
            tid
            for tid, t in self.tracks.items()
            if t.coasting_frames > self.max_coast_frames
        ]
        for tid in dead:
            del self.tracks[tid]

        # 7) 轨迹级合并：消灭"整杯框 + 半截框"养出的影子轨迹
        self._merge_tracks()

        return sorted(self.tracks.values(), key=lambda t: -t.confidence)

    # ---------- 防御 A：同类轨迹合并 ----------

    @staticmethod
    def _containment(a: Sequence[float], b: Sequence[float]) -> float:
        """交集面积 / 较小框面积。完全包含时 = 1.0。"""
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
        ih = max(0.0, min(ay2, by2) - max(ay1, by1))
        inter = iw * ih
        if inter <= 0:
            return 0.0
        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        return inter / min(area_a, area_b)

    def _merge_tracks(self) -> None:
        """同类两条轨迹持续重叠 → 同一物体，保留证据更强的一条。

        判定信号（满足其一即"本帧重叠"）：
          IoU >= merge_iou                —— 两框互为彼此
          containment >= merge_contain    —— 小框几乎整个躺在大框里
        后者专治"半截框"：IoU 上不去（0.47），但包含率接近 1。

        为什么要 merge_streak 持续帧确认：
          两个真实杯子挨在一起或交叉穿行时也会瞬时重叠，
          一发现重叠就合并会误删。影子框的特征是"永远重叠"
          （同一物体的两个部件），真目标的重叠是暂时的。
          连续重叠 merge_streak 帧才合并，区分这两种情况。
        """
        by_obj: dict[str, list[TrackedObject]] = {}
        for t in self.tracks.values():
            by_obj.setdefault(t.object_id, []).append(t)

        overlapping_pairs: set[tuple[int, int]] = set()
        victims: set[int] = set()
        for group in by_obj.values():
            group.sort(key=lambda t: -t.confidence)
            for i, a in enumerate(group):
                if a.track_id in victims:
                    continue
                for b in group[i + 1:]:
                    if b.track_id in victims:
                        continue
                    iou = _iou(a.bbox, b.bbox)
                    cont = self._containment(a.bbox, b.bbox)
                    if iou >= self.merge_iou or cont >= self.merge_contain:
                        pair = (a.track_id, b.track_id)
                        overlapping_pairs.add(pair)
                        streak = self._streaks.get(pair, 0) + 1
                        self._streaks[pair] = streak
                        if streak >= self.merge_streak:
                            a.hits += b.hits // 2
                            victims.add(b.track_id)
                            self._streaks.pop(pair, None)
        # 本帧没重叠的旧计数清零（重叠断了就不算影子）
        for pair in list(self._streaks):
            if pair not in overlapping_pairs:
                del self._streaks[pair]
        for tid in victims:
            self.tracks.pop(tid, None)

    # ---------- 防御 B：补充框不起新轨迹 ----------

    def _is_supplementary(self, det: Detection) -> bool:
        """检测框中心落在某条同物体轨迹框内 → 它是既有目标的局部检出。"""
        cx, cy = float(det.center_xy[0]), float(det.center_xy[1])
        for t in self.tracks.values():
            if t.object_id != det.object_id:
                continue
            x1, y1, x2, y2 = t.bbox
            if x1 <= cx <= x2 and y1 <= cy <= y2:
                return True
        return False

    # ---------- 关联 ----------

    def _associate(
        self,
        dets: list[Detection],
        predicted: dict[int, list[float]],
    ) -> tuple[list[tuple[int, Detection]], list[int], list[Detection]]:
        cands: list[tuple[float, float, int, int]] = []  # (iou, conf, tid, di)
        for tid, t in self.tracks.items():
            for di, d in enumerate(dets):
                if d.object_id != t.object_id:
                    continue  # 关键先验：绝不跨类关联
                iou = _iou(predicted[tid], d.bbox_xyxy)
                if iou >= self.match_iou:
                    cands.append((iou, d.confidence, tid, di))

        cands.sort(key=lambda c: (-c[0], -c[1]))
        used_t: set[int] = set()
        used_d: set[int] = set()
        matched: list[tuple[int, Detection]] = []
        for iou, conf, tid, di in cands:
            if tid in used_t or di in used_d:
                continue
            used_t.add(tid)
            used_d.add(di)
            matched.append((tid, dets[di]))

        unmatched_tracks = [tid for tid in self.tracks if tid not in used_t]
        unmatched_dets = [d for i, d in enumerate(dets) if i not in used_d]
        return matched, unmatched_tracks, unmatched_dets

    # ---------- 观测更新 / 新建 ----------

    def _update_from_obs(
        self,
        t: TrackedObject,
        det: Detection,
        pred_bbox: list[float],
        frame_idx: int,
        dt: float,
    ) -> None:
        obs_c = [float(c) for c in det.center_xy]
        pred_c = [
            (pred_bbox[0] + pred_bbox[2]) / 2.0,
            (pred_bbox[1] + pred_bbox[3]) / 2.0,
        ]

        if t.coasting_frames >= 5:
            # 从滑行中抢回：用"整段遮挡的平均速度"回归。
            # 不能把单帧观测残差除以 dt——dt=1/30 会把半截框带来的
            # 55px 中心跳变放大成 1650 px/s，恒速外推直接把轨迹甩出画面
            # （demo 实测踩坑：静止杯的轨迹被"甩"到 x=2777 还 confirmed）。
            #
            # 冷却 5 帧的原因：抢回后的下一帧若再遇半截框，
            # coasting 归零会让每次观测都"重新回归"，污染叠加。
            # 5 帧内观测只更新位置、不动速度（位置本来就是对的）。
            total_dt = (t.coasting_frames + 1) * dt
            start_c = t.trail[-1] if t.trail else pred_c
            v_avg = [(obs_c[i] - start_c[i]) / total_dt for i in range(2)]
            sp_new = math.hypot(v_avg[0], v_avg[1])
            sp_old = math.hypot(t.vx, t.vy)
            # 方向校验：回归速度若与遮挡前速度反向且明显更快，
            # 大概率是错配（被别的物体/半截框拉走），只弱吸收
            dot = t.vx * v_avg[0] + t.vy * v_avg[1]
            alpha = 0.3 if (dot < 0 and sp_new > sp_old) else 0.6
            t.vx = t.vx + (v_avg[0] - t.vx) * alpha
            t.vy = t.vy + (v_avg[1] - t.vy) * alpha
        elif t.coasting_frames == 0:
            # 残差速度：观测相对恒速预测的偏差，阻尼收敛、无放大
            residual_v = [(obs_c[i] - pred_c[i]) / dt for i in range(2)]
            t.vx = t.vx + residual_v[0] * 0.4
            t.vy = t.vy + residual_v[1] * 0.4
        # else: 抢回后 1~4 帧 → 观测只更新位置，速度保持冻结

        # 速度钳制：双保险
        sp = math.hypot(t.vx, t.vy)
        if sp > self.max_speed:
            k = self.max_speed / sp
            t.vx *= k
            t.vy *= k

        t.bbox = [float(v) for v in det.bbox_xyxy]
        t.center = obs_c
        t.width = t.bbox[2] - t.bbox[0]
        t.height = t.bbox[3] - t.bbox[1]
        t.confidence = det.confidence
        t.hits += 1
        t.coasting_frames = 0
        t.is_predicted = False
        t.confirmed = t.hits >= self.min_hits
        t.last_seen_frame = frame_idx
        t.trail.append([int(v) for v in obs_c])
        if len(t.trail) > 60:
            t.trail.pop(0)

    def _spawn(self, det: Detection, frame_idx: int) -> None:
        x1, y1, x2, y2 = det.bbox_xyxy
        t = TrackedObject(
            object_id=det.object_id,
            name_cn=det.name_cn,
            track_id=self._next_id,
            bbox=[float(v) for v in det.bbox_xyxy],
            center=[float(c) for c in det.center_xy],
            width=x2 - x1,
            height=y2 - y1,
            confidence=det.confidence,
            first_frame=frame_idx,
            last_seen_frame=frame_idx,
            hits=1,
            trail=[[det.center_xy[0], det.center_xy[1]]],
        )
        self.tracks[self._next_id] = t
        self._next_id += 1


class TrackerFrameResult:
    """单帧追踪输出。"""

    def __init__(
        self,
        frame_idx: int,
        detections: list[Detection],
        tracks: list[TrackedObject],
        infer_ms: float,
        ran_detection: bool,
    ) -> None:
        self.frame_idx = frame_idx
        self.detections = detections
        self.tracks = tracks
        self.infer_ms = infer_ms
        self.ran_detection = ran_detection

    def tracks_by_object(self, object_id: str) -> list[TrackedObject]:
        return [t for t in self.tracks if t.object_id == object_id]

    def count_confirmed(self, object_id: str) -> int:
        """已确认轨迹数 —— 这就是"桌上现在有几个杯子"的答案。"""
        return sum(
            1
            for t in self.tracks
            if t.object_id == object_id and t.confirmed
        )


class ObjectTracker:
    """YOLO-World 检测 + SimpleTracker 身份维持，检测/追踪分频。

    detect_every=N：每 N 帧才跑一次检测（方案第 17 节
    "本地检测低频跑、身份靠时间补"的直接实现），
    其余帧纯靠恒速预测滑行。
    """

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
        max_coast_frames: int = 75,
        min_hits: int = 3,
        merge_iou: float = 0.55,
        merge_contain: float = 0.75,
        fps: float = 30.0,
        verbose: bool = False,
    ) -> None:
        self.detector = ObjectDetector(
            specs,
            model_size=model_size,
            device=device,
            imgsz=imgsz,
            dup_iou=dup_iou,
            verbose=verbose,
        )
        self.tracker = SimpleTracker(
            match_iou=match_iou,
            max_coast_frames=max_coast_frames,
            min_hits=min_hits,
            merge_iou=merge_iou,
            merge_contain=merge_contain,
        )
        self.detect_every = max(1, detect_every)
        self.fps = fps
        self.frame_idx = 0
        self.infer_times: list[float] = []
        self._color_of = {sp.id: sp.color for sp in self.detector.specs}

    def process(self, frame: np.ndarray) -> TrackerFrameResult:
        ran = self.frame_idx % self.detect_every == 0
        dets: list[Detection] = []
        infer_ms = 0.0
        if ran:
            res = self.detector.detect(frame)
            dets = res.detections
            infer_ms = res.infer_ms
            self.infer_times.append(infer_ms)
        tracks = self.tracker.step(dets, self.frame_idx, self.fps)
        out = TrackerFrameResult(
            self.frame_idx, dets, tracks, infer_ms, ran
        )
        self.frame_idx += 1
        return out

    # ---------- 可视化 ----------

    def draw(self, frame: np.ndarray, result: TrackerFrameResult) -> np.ndarray:
        canvas = frame.copy()

        for t in result.tracks:
            color = self._color_of.get(t.object_id, (0, 200, 0))
            x1, y1, x2, y2 = (int(v) for v in t.bbox)
            if not t.confirmed:
                th = 1
            elif t.is_predicted:
                th = 2
            else:
                th = 2
            # 预测态用虚线效果：四角只画一段
            if t.is_predicted:
                Lx, Ly = (x2 - x1) // 4, (y2 - y1) // 4
                for (ax, ay, bx, by) in [
                    (x1, y1, x1 + Lx, y1), (x1, y1, x1, y1 + Ly),
                    (x2, y1, x2 - Lx, y1), (x2, y1, x2, y1 + Ly),
                    (x1, y2, x1 + Lx, y2), (x1, y2, x1, y2 - Ly),
                    (x2, y2, x2 - Lx, y2), (x2, y2, x2, y2 - Ly),
                ]:
                    cv2.line(canvas, (ax, ay), (bx, by), color, th)
            else:
                cv2.rectangle(canvas, (x1, y1), (x2, y2), color, th)

            # 轨迹尾巴
            for i in range(1, len(t.trail)):
                if t.trail[i - 1][0] < 0:
                    continue
                cv2.line(canvas, tuple(t.trail[i - 1]), tuple(t.trail[i]), color, 1,
                         cv2.LINE_AA)

            tag = "C" if t.confirmed else "T"
            if t.is_predicted:
                tag = f"P{t.coasting_frames}"
            label = f"{t.name_cn}#{t.track_id} {tag} {t.confidence:.2f}"
            (tw, thh), base = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            ty = max(y1 - thh - base - 2, 0)
            cv2.rectangle(canvas, (x1, ty), (x1 + tw + 4, ty + thh + base + 2),
                          color, -1)
            cv2.putText(canvas, label, (x1 + 2, ty + thh),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                        cv2.LINE_AA)

        return canvas
