#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vision.object_tracker import ObjectTracker
from vision.object_detector import (
    ObjectSpec,
    specs_from_config,
)
from vision.hand_detector import HandDetector
from vision.hand_types import HandObservation
from perception_serializer import serialize_perception
from activities.activity_engine import ActivityEngine
from activities.activity_understanding import ActivityUnderstanding
from config.model_config import (
    VLM_COOLDOWN,
    VLM_MODEL,
    VLM_TRIGGER_EVENTS,
)


# ============================================================
# 默认模型
# ============================================================

HAND_MODEL_PATH = (
    ROOT / "models" / "hand_landmarker.task"
)


# ============================================================
# 配置
# ============================================================

def resolve_project_path(value: str | Path) -> Path:
    """相对路径统一以项目根目录为基准。"""
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def load_activity_config(config_path: str | Path) -> dict:
    """加载并返回 activity 配置主体。"""
    path = resolve_project_path(config_path)
    if not path.exists():
        sys.exit(f"配置文件不存在: {path}")

    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    body = raw.get("activity", raw)
    if not isinstance(body, dict):
        sys.exit(f"配置文件格式错误: {path}")
    return body


def build_specs(args, cfg_body: dict) -> list[ObjectSpec]:
    """从 YAML 构建物体规格；--prompt 仅作为临时覆盖入口。"""
    if args.prompt:
        return [
            ObjectSpec(
                id=p.replace(" ", "_"),
                name_cn=p,
                prompts=[p],
                conf=0.02,
            )
            for p in args.prompt
        ]

    specs = specs_from_config(cfg_body)
    if not specs:
        sys.exit(f"配置里没有可用的 objects: {args.config}")
    return specs


def cfg_value(args_value, config_value, default):
    """CLI 显式传值优先，否则使用 YAML，最后使用代码兜底值。"""
    return default if args_value is None and config_value is None else (
        config_value if args_value is None else args_value
    )


# ============================================================
# 绘制手部
# ============================================================

def draw_hand(
    canvas: np.ndarray,
    hand: HandObservation,
) -> None:

    # --------------------------------------------------------
    # 21 个关键点
    # --------------------------------------------------------

    points = hand.landmarks

    if len(points) != 21:
        return

    # MediaPipe 手部骨架连接
    connections = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),

        (0, 5),
        (5, 6),
        (6, 7),
        (7, 8),

        (5, 9),
        (9, 10),
        (10, 11),
        (11, 12),

        (9, 13),
        (13, 14),
        (14, 15),
        (15, 16),

        (13, 17),
        (17, 18),
        (18, 19),
        (19, 20),

        (0, 17),
    ]

    # --------------------------------------------------------
    # 骨架
    # --------------------------------------------------------

    for start, end in connections:

        p1 = points[start]
        p2 = points[end]

        x1, y1 = int(p1[0]), int(p1[1])
        x2, y2 = int(p2[0]), int(p2[1])

        cv2.line(
            canvas,
            (x1, y1),
            (x2, y2),
            (255, 180, 0),
            2,
            cv2.LINE_AA,
        )

    # --------------------------------------------------------
    # 关键点
    # --------------------------------------------------------

    for index, point in enumerate(points):

        x = int(point[0])
        y = int(point[1])

        radius = 5 if index == 0 else 4

        cv2.circle(
            canvas,
            (x, y),
            radius,
            (0, 255, 255),
            -1,
            cv2.LINE_AA,
        )

    # --------------------------------------------------------
    # bbox
    # --------------------------------------------------------

    x1, y1, x2, y2 = (
        int(v)
        for v in hand.bbox
    )

    cv2.rectangle(
        canvas,
        (x1, y1),
        (x2, y2),
        (255, 180, 0),
        2,
    )

    # --------------------------------------------------------
    # wrist
    # --------------------------------------------------------

    wrist_x = int(hand.wrist[0])
    wrist_y = int(hand.wrist[1])

    cv2.circle(
        canvas,
        (wrist_x, wrist_y),
        8,
        (0, 0, 255),
        -1,
        cv2.LINE_AA,
    )

    # --------------------------------------------------------
    # 标签
    # --------------------------------------------------------

    label = (
        f"{hand.hand_id} Hand "
        f"{hand.confidence:.2f}"
    )

    (tw, th), baseline = cv2.getTextSize(
        label,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        2,
    )

    tx = x1
    ty = max(
        y1 - 8,
        th + baseline + 5,
    )

    cv2.rectangle(
        canvas,
        (
            tx,
            ty - th - baseline - 4,
        ),
        (
            tx + tw + 8,
            ty + 4,
        ),
        (255, 180, 0),
        -1,
    )

    cv2.putText(
        canvas,
        label,
        (
            tx + 4,
            ty,
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


# ============================================================
# 计算手和物体的距离
# ============================================================

def point_to_bbox_distance(
    point: tuple[float, float],
    bbox: list[float],
) -> float:

    px, py = point

    x1, y1, x2, y2 = bbox

    dx = max(
        x1 - px,
        0.0,
        px - x2,
    )

    dy = max(
        y1 - py,
        0.0,
        py - y2,
    )

    return float(
        (dx * dx + dy * dy) ** 0.5
    )


# ============================================================
# 显示信息
# ============================================================

def draw_info_panel(
    canvas: np.ndarray,
    hands: list[HandObservation],
    tracks,
    fps: float,
) -> None:

    x = 20
    y = 35

    font = cv2.FONT_HERSHEY_SIMPLEX

    # --------------------------------------------------------
    # 标题
    # --------------------------------------------------------

    cv2.putText(
        canvas,
        "PiPi Vision 2.0",
        (x, y),
        font,
        0.9,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    y += 35

    cv2.putText(
        canvas,
        f"FPS: {fps:.1f}",
        (x, y),
        font,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    y += 30

    # --------------------------------------------------------
    # 手部
    # --------------------------------------------------------

    cv2.putText(
        canvas,
        f"Hands: {len(hands)}",
        (x, y),
        font,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    y += 28

    for hand in hands:

        wx, wy = hand.wrist

        text = (
            f"{hand.hand_id}: "
            f"wrist=({int(wx)}, {int(wy)})"
        )

        cv2.putText(
            canvas,
            text,
            (x, y),
            font,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        y += 25

    # --------------------------------------------------------
    # 纸杯
    # --------------------------------------------------------

    confirmed_tracks = [
        t
        for t in tracks
        if t.confirmed
    ]

    y += 8

    cv2.putText(
        canvas,
        (
            f"Objects: "
            f"{len(confirmed_tracks)} confirmed"
        ),
        (x, y),
        font,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    y += 28

    for track in confirmed_tracks:

        speed = track.speed_px_s

        state = (
            "预测中"
            if track.is_predicted
            else "检测到"
        )

        text = (
            f"{track.name_cn} "
            f"#{track.track_id} "
            f"{state} "
            f"{speed:.0f}px/s"
        )

        cv2.putText(
            canvas,
            text,
            (x, y),
            font,
            0.52,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        y += 24


def draw_activity_panel(
    canvas: np.ndarray,
    engine: ActivityEngine,
    events: list[dict],
) -> None:
    x = 20
    y = canvas.shape[0] - 105
    font = cv2.FONT_HERSHEY_SIMPLEX

    cv2.putText(
        canvas,
        f"Activity: {engine.activity_name}",
        (x, y),
        font,
        0.62,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    y += 25
    cv2.putText(
        canvas,
        f"State: {engine.last_state} | Engine: {engine.total_event_count} events",
        (x, y),
        font,
        0.52,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    y += 25
    visible_events = [
        event
        for event in events
        if float(event["event"].get("confidence", 0.0)) > ACTIVITY_DISPLAY_MIN_CONFIDENCE
    ]

    if visible_events:
        latest = visible_events[-1]["event"]
        text = f"Event: {latest['type']}  conf={latest['confidence']:.2f}"
    else:
        text = "Event: 无新事件"
    cv2.putText(
        canvas,
        text,
        (x, y),
        font,
        0.55,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


# ============================================================
# Activity 显示置信度阈值
# 仅控制终端/OpenCV 面板显示，不影响 ActivityEngine 内部事件生成。
# ============================================================
ACTIVITY_DISPLAY_MIN_CONFIDENCE = 0.01


# ============================================================
# 主程序
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "PiPi Vision 2.0 "
            "YOLO-World + Tracker + MediaPipe Hand"
        )
    )

    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="摄像头编号，默认 0",
    )

    parser.add_argument(
        "--config",
        "-c",
        default=(
            "activities/configs/"
            "stacking_coins.yaml"
        ),
        help="活动 YAML 配置",
    )

    parser.add_argument(
        "--prompt",
        "-p",
        nargs="+",
        help="直接指定英文物体 prompt",
    )

    parser.add_argument(
        "--size",
        default=None,
        choices=list("nsmlx"),
        help="YOLO-World 模型规模；默认读取 YAML vision.object_detector.model_size",
    )

    parser.add_argument(
        "--imgsz",
        type=int,
        default=None,
        help="YOLO 输入尺寸；默认读取 YAML vision.object_detector.imgsz",
    )

    parser.add_argument(
        "--device",
        default=None,
        help="推理设备；默认读取 YAML vision.object_detector.device",
    )

    parser.add_argument(
        "--dup-iou",
        type=float,
        default=None,
        help="检测框去重 IoU；默认读取 YAML vision.object_detector.dup_iou",
    )

    parser.add_argument(
        "--detect-every",
        type=int,
        default=None,
        help="每 N 帧运行一次 YOLO；默认读取 YAML vision.tracker.detect_every",
    )

    parser.add_argument(
        "--match-iou",
        type=float,
        default=None,
        help="轨迹匹配 IoU；默认读取 YAML vision.tracker.match_iou",
    )

    parser.add_argument(
        "--max-coast",
        type=int,
        default=None,
        help="最大滑行帧数；默认读取 YAML vision.tracker.max_coast_frames",
    )

    parser.add_argument(
        "--min-hits",
        type=int,
        default=None,
        help="轨迹确认所需命中次数；默认读取 YAML vision.tracker.min_hits",
    )

    parser.add_argument(
        "--hand-model",
        default=None,
        help="手部模型路径；默认读取 YAML vision.hand_detector.model_path",
    )

    parser.add_argument(
        "--no-hand",
        action="store_true",
        help="关闭 MediaPipe 手部检测",
    )

    parser.add_argument(
        "--no-show",
        action="store_true",
        help="不显示 OpenCV 窗口",
    )

    parser.add_argument(
        "--activity-rate",
        type=float,
        default=None,
        help="Activity Engine 输入频率；默认读取 YAML runtime.activity_rate_hz",
    )

    parser.add_argument(
        "--activity-min-confidence",
        type=float,
        default=ACTIVITY_DISPLAY_MIN_CONFIDENCE,
        help="Activity 显示的最低置信度；默认 0.60。仅过滤显示，不影响 ActivityEngine。",
    )

    parser.add_argument(
        "--activity-jsonl",
        default=None,
        help="输出 Activity Engine 输入 Perception JSONL；不指定则不落盘",
    )

    parser.add_argument(
        "--event-jsonl",
        default=None,
        help="输出 Activity Engine 事件 JSONL；不指定则不落盘",
    )

    parser.add_argument(
        "--no-vlm",
        action="store_true",
        help="关闭 Qwen-VL 高级活动理解",
    )

    parser.add_argument(
        "--vlm-model",
        default=None,
        help="临时覆盖 Qwen-VL 模型名称；默认读取 config/model_config.py",
    )

    parser.add_argument(
        "--vlm-cooldown",
        type=float,
        default=None,
        help="临时覆盖两次 Qwen-VL 请求之间的最小间隔；默认读取 config/model_config.py",
    )

    parser.add_argument(
        "--vlm-events",
        default=None,
        help="临时覆盖触发 Qwen-VL 的 ActivityEngine 事件，逗号分隔；默认读取 config/model_config.py",
    )

    parser.add_argument(
        "--vlm-jsonl",
        default=None,
        help="输出 Qwen-VL 结果 JSONL；不指定则不落盘",
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Qwen-VL 配置统一从 config/model_config.py 读取。
    # CLI 参数仍然保留，用于临时测试时覆盖配置。
    # --------------------------------------------------------
    args.vlm_model = args.vlm_model or VLM_MODEL
    args.vlm_cooldown = (
        VLM_COOLDOWN if args.vlm_cooldown is None else args.vlm_cooldown
    )
    args.vlm_events = args.vlm_events or ",".join(sorted(VLM_TRIGGER_EVENTS))

    args.activity_min_confidence = max(0.0, min(1.0, float(args.activity_min_confidence)))

    cfg_body = load_activity_config(args.config)
    detector_cfg = cfg_body.get("vision", {}).get("object_detector", {}) or {}
    tracker_cfg = cfg_body.get("vision", {}).get("tracker", {}) or {}
    hand_cfg = cfg_body.get("vision", {}).get("hand_detector", {}) or {}
    runtime_cfg = cfg_body.get("runtime", {}) or {}

    args.size = cfg_value(args.size, detector_cfg.get("model_size"), "s")
    args.imgsz = int(cfg_value(args.imgsz, detector_cfg.get("imgsz"), 640))
    args.device = cfg_value(args.device, detector_cfg.get("device"), "auto")
    args.dup_iou = float(cfg_value(args.dup_iou, detector_cfg.get("dup_iou"), 0.65))
    args.detect_every = int(cfg_value(args.detect_every, tracker_cfg.get("detect_every"), 1))
    args.match_iou = float(cfg_value(args.match_iou, tracker_cfg.get("match_iou"), 0.25))
    args.max_coast = int(cfg_value(args.max_coast, tracker_cfg.get("max_coast_frames"), 5))
    args.min_hits = int(cfg_value(args.min_hits, tracker_cfg.get("min_hits"), 3))
    args.hand_model = str(resolve_project_path(cfg_value(args.hand_model, hand_cfg.get("model_path"), HAND_MODEL_PATH)))
    args.activity_rate = float(cfg_value(args.activity_rate, runtime_cfg.get("activity_rate_hz"), 10.0))

    # --------------------------------------------------------
    # Object Specs
    # --------------------------------------------------------

    specs = build_specs(args, cfg_body)

    # --------------------------------------------------------
    # 摄像头
    # --------------------------------------------------------

    cap = cv2.VideoCapture(
        args.camera
    )

    if not cap.isOpened():

        sys.exit(
            f"无法打开摄像头: {args.camera}\n"
            "请检查 macOS 摄像头权限。"
        )

    ok, first_frame = cap.read()

    if not ok or first_frame is None:

        cap.release()

        sys.exit(
            "无法读取摄像头第一帧。"
        )

    height, width = first_frame.shape[:2]

    camera_fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    if (
        camera_fps is None
        or camera_fps <= 1
    ):
        camera_fps = 30.0

    # --------------------------------------------------------
    # Object Tracker
    # --------------------------------------------------------

    tracker = ObjectTracker(
        specs,
        model_size=args.size,
        device=args.device,
        imgsz=args.imgsz,
        dup_iou=args.dup_iou,
        detect_every=args.detect_every,
        match_iou=args.match_iou,
        max_coast_frames=args.max_coast,
        min_hits=args.min_hits,
        merge_iou=float(tracker_cfg.get("merge_iou", 0.60)),
        merge_contain=float(tracker_cfg.get("merge_contain", 0.80)),
        merge_streak=int(tracker_cfg.get("merge_streak", 3)),
        speed_history_size=int(tracker_cfg.get("speed_history_size", 5)),
        speed_ema_alpha=float(tracker_cfg.get("speed_ema_alpha", 0.35)),
        max_center_jump_ratio=float(tracker_cfg.get("max_center_jump_ratio", 3.0)),
        low_confidence_threshold=float(tracker_cfg.get("low_confidence_threshold", 0.10)),
        coast_speed_decay=float(tracker_cfg.get("coast_speed_decay", 0.80)),
        fps=camera_fps,
        verbose=True,
    )

    # --------------------------------------------------------
    # Hand Detector
    # --------------------------------------------------------

    hand_detector = None

    if not args.no_hand:

        hand_detector = HandDetector(
            model_path=args.hand_model,
            num_hands=int(hand_cfg.get("num_hands", 2)),
            min_hand_detection_confidence=float(hand_cfg.get("min_hand_detection_confidence", 0.5)),
            min_hand_presence_confidence=float(hand_cfg.get("min_hand_presence_confidence", 0.5)),
            min_tracking_confidence=float(hand_cfg.get("min_tracking_confidence", 0.5)),
        )

    # --------------------------------------------------------
    # Activity Engine
    # --------------------------------------------------------
    cfg_path = resolve_project_path(args.config)
    cfg_raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    cfg_body = cfg_raw.get("activity", cfg_raw)
    activity_id = str(cfg_body.get("id", cfg_path.stem))
    activity_name = str(cfg_body.get("name", activity_id))

    activity_rate = max(0.1, float(args.activity_rate))
    activity_interval = 1.0 / activity_rate
    window_seconds = max(0.5, float(runtime_cfg.get("window_seconds", 2.0)))
    activity_engine = ActivityEngine(
        cfg_path,
        window_seconds=window_seconds,
    )

    perception_file = None
    event_file = None
    if args.activity_jsonl:
        perception_path = Path(args.activity_jsonl)
        perception_path.parent.mkdir(parents=True, exist_ok=True)
        perception_file = perception_path.open("w", encoding="utf-8")
    if args.event_jsonl:
        event_path = Path(args.event_jsonl)
        event_path.parent.mkdir(parents=True, exist_ok=True)
        event_file = event_path.open("w", encoding="utf-8")

    # --------------------------------------------------------
    # Qwen-VL Activity Understanding
    # --------------------------------------------------------
    vlm = None
    vlm_file = None

    if not args.no_vlm:
        try:
            trigger_events = {
                item.strip().upper()
                for item in str(args.vlm_events).split(",")
                if item.strip()
            }
            vlm = ActivityUnderstanding(
                activity_id=activity_id,
                activity_name=activity_name,
                model=args.vlm_model,
                cooldown_seconds=float(args.vlm_cooldown),
                trigger_events=trigger_events,
            )
            print(
                f"Qwen-VL: ON, model={args.vlm_model}, "
                f"cooldown={args.vlm_cooldown:.1f}s, events={sorted(trigger_events)}"
            )
        except Exception as exc:
            print(f"[VLM] 初始化失败，已自动关闭: {exc}")

    if args.vlm_jsonl:
        vlm_path = Path(args.vlm_jsonl)
        vlm_path.parent.mkdir(parents=True, exist_ok=True)
        vlm_file = vlm_path.open("w", encoding="utf-8")

    last_activity_time = 0.0
    last_events = []
    last_vlm_result = None

    # --------------------------------------------------------
    # 运行
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("PiPi Vision 2.0")
    print("=" * 60)
    print(
        f"Camera: {args.camera}"
    )
    print(
        f"Resolution: "
        f"{width}x{height}"
    )
    print(
        f"FPS: {camera_fps:.1f}"
    )
    print(
        f"Hand detector: "
        f"{'ON' if hand_detector else 'OFF'}"
    )
    print(
        f"YOLO: size={args.size}, imgsz={args.imgsz}, device={args.device}"
    )
    print(
        f"Tracker: detect_every={args.detect_every}, match_iou={args.match_iou:.2f}, "
        f"max_coast={args.max_coast}, min_hits={args.min_hits}"
    )
    print(
        f"Activity Engine: {activity_rate:.1f}Hz, window={window_seconds:.1f}s, "
        f"near={activity_engine.near_distance:.0f}px"
    )
    print(
        "按 q 或 ESC 退出"
    )
    print("=" * 60)
    print()

    frame_index = 0

    last_time = time.perf_counter()

    display_fps = camera_fps

    try:

        while True:

            if frame_index == 0:

                frame = first_frame

            else:

                ok, frame = cap.read()

                if not ok or frame is None:
                    break

            # ------------------------------------------------
            # 镜像
            # ------------------------------------------------

            frame = cv2.flip(
                frame,
                1,
            )

            # ------------------------------------------------
            # YOLO + Tracker
            # ------------------------------------------------

            result = tracker.process(
                frame
            )

            canvas = tracker.draw(
                frame,
                result,
            )

            # ------------------------------------------------
            # MediaPipe Hand
            # ------------------------------------------------

            hands: list[
                HandObservation
            ] = []

            if hand_detector:

                # VIDEO 模式 timestamp 必须递增
                timestamp_ms = int(
                    frame_index
                    * 1000.0
                    / camera_fps
                )

                hands = hand_detector.detect(
                    frame,
                    timestamp_ms,
                )

                for hand in hands:

                    draw_hand(
                        canvas,
                        hand,
                    )

            # ------------------------------------------------
            # 手 -> 纸杯关系
            # ------------------------------------------------

            confirmed_tracks = [
                t
                for t in result.tracks
                if t.confirmed
            ]

            for hand in hands:

                for track in confirmed_tracks:

                    distance = (
                        point_to_bbox_distance(
                            hand.wrist,
                            track.bbox,
                        )
                    )

                    # 距离比较近时画一条辅助线
                    if distance <= activity_engine.near_distance:

                        hx, hy = (
                            int(hand.wrist[0]),
                            int(hand.wrist[1]),
                        )

                        cx, cy = (
                            int(track.center[0]),
                            int(track.center[1]),
                        )

                        cv2.line(
                            canvas,
                            (hx, hy),
                            (cx, cy),
                            (255, 255, 0),
                            1,
                            cv2.LINE_AA,
                        )

                        mid_x = (
                            hx + cx
                        ) // 2

                        mid_y = (
                            hy + cy
                        ) // 2

                        cv2.putText(
                            canvas,
                            f"{distance:.0f}px",
                            (
                                mid_x,
                                mid_y,
                            ),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.45,
                            (255, 255, 0),
                            1,
                            cv2.LINE_AA,
                        )

            # ------------------------------------------------
            # FPS
            # ------------------------------------------------

            now = time.perf_counter()

            dt = now - last_time

            last_time = now

            if dt > 0:

                instant_fps = 1.0 / dt

                display_fps = (
                    display_fps * 0.9
                    + instant_fps * 0.1
                )

            # ------------------------------------------------
            # Activity Engine：10Hz 标准化感知输入
            # ------------------------------------------------
            now_activity = time.perf_counter()
            if now_activity - last_activity_time >= activity_interval:
                perception = serialize_perception(
                    frame_index=frame_index,
                    timestamp=time.time(),
                    frame_width=width,
                    frame_height=height,
                    fps=display_fps,
                    hands=hands,
                    tracker_result=result,
                    activity_id=activity_id,
                    activity_name=activity_name,
                )
                last_activity_time = now_activity

                if perception_file is not None:
                    perception_file.write(
                        json.dumps(perception, ensure_ascii=False) + "\n"
                    )

                events = activity_engine.update(perception)
                last_events = events

                for event in events:
                    event_type = event["event"]["type"]
                    confidence = float(event["event"].get("confidence", 0.0))
                    track_id = event["event"].get("object", {}).get("track_id")
                    object_text = f"  物体#{track_id}" if track_id is not None else ""

                    # ActivityEngine 仍然保留并输出全部事件。
                    # 这里只过滤终端显示：只有 confidence > 0.60 才显示。
                    if confidence > args.activity_min_confidence:
                        print(
                            f"[Activity] {event_type}{object_text} "
                            f"置信度={confidence:.2f}"
                        )

                    # JSONL 仍保存全部事件，方便后续 VLM / Agent / 调试使用。
                    if event_file is not None:
                        event_file.write(
                            json.dumps(event, ensure_ascii=False) + "\n"
                        )

                    # ----------------------------------------------------
                    # 关键通用事件 -> Qwen-VL
                    # ActivityEngine 不负责 STACKED / COMPLETED 等活动语义。
                    # VLM 在后台线程中分析当前帧，避免阻塞摄像头循环。
                    # ----------------------------------------------------
                    if vlm is not None:
                        submitted = vlm.submit(
                            canvas,
                            event,
                            perception,
                        )
                        if submitted:
                            print(f"[VLM] 已提交 {event_type} -> Qwen-VL")

            # ----------------------------------------------------
            # 读取后台 Qwen-VL 结果
            # ----------------------------------------------------
            if vlm is not None:
                for vlm_result in vlm.poll_results():
                    last_vlm_result = vlm_result

                    if vlm_result.get("ok"):
                        result_data = vlm_result.get("result", {})
                        raw_text = result_data.get("raw_text", "")
                        parsed = result_data.get("parsed")

                        print("\n[VLM] Qwen-VL 返回:")
                        print(raw_text)
                        if parsed is None:
                            print("[VLM] 警告：返回内容未解析成 JSON。")
                        else:
                            print(
                                "[VLM JSON] "
                                + json.dumps(parsed, ensure_ascii=False)
                            )

                        if vlm_file is not None:
                            vlm_file.write(
                                json.dumps(vlm_result, ensure_ascii=False) + "\n"
                            )
                            vlm_file.flush()
                    else:
                        print(f"[VLM] 请求失败: {vlm_result.get('error', 'unknown error')}")
                        if vlm_file is not None:
                            vlm_file.write(
                                json.dumps(vlm_result, ensure_ascii=False) + "\n"
                            )
                            vlm_file.flush()

            # ------------------------------------------------
            # 信息面板
            # ------------------------------------------------

            draw_info_panel(
                canvas,
                hands,
                result.tracks,
                display_fps,
            )

            draw_activity_panel(
                canvas,
                activity_engine,
                last_events,
            )

            # ------------------------------------------------
            # 显示
            # ------------------------------------------------

            if not args.no_show:

                cv2.imshow(
                    "PiPi Vision 2.0",
                    canvas,
                )

                key = (
                    cv2.waitKey(1)
                    & 0xFF
                )

                if key in (
                    ord("q"),
                    27,
                ):
                    break

            frame_index += 1

    except KeyboardInterrupt:

        print(
            "\n收到 Ctrl+C，正在退出..."
        )

    finally:

        if vlm is not None:
            vlm.close()

        if perception_file is not None:
            perception_file.close()

        if event_file is not None:
            event_file.close()

        if vlm_file is not None:
            vlm_file.close()

        cap.release()

        if hand_detector:

            hand_detector.close()

        cv2.destroyAllWindows()

    print()
    print(
        f"程序结束，共处理 "
        f"{frame_index} 帧"
    )


if __name__ == "__main__":
    main()