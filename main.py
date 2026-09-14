#!/usr/bin/env python

from __future__ import annotations

import argparse
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


# ============================================================
# 默认模型
# ============================================================

HAND_MODEL_PATH = (
    ROOT / "models" / "hand_landmarker.task"
)


# ============================================================
# 配置
# ============================================================

def build_specs(args) -> list[ObjectSpec]:

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

    if not args.config:
        sys.exit(
            "必须提供 --config <yaml> "
            "或 --prompt <英文短语...>"
        )

    cfg_path = Path(args.config)

    if not cfg_path.exists():
        sys.exit(
            f"配置文件不存在: {cfg_path}"
        )

    cfg = yaml.safe_load(
        cfg_path.read_text(
            encoding="utf-8"
        )
    ) or {}

    body = cfg.get(
        "activity",
        cfg,
    )

    specs = specs_from_config(body)

    if not specs:
        sys.exit(
            f"配置里没有可用的 objects: {cfg_path}"
        )

    return specs


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
            "paper_cup.yaml"
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
        default="s",
        choices=list("nsmlx"),
    )

    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
    )

    parser.add_argument(
        "--device",
        default="auto",
    )

    parser.add_argument(
        "--dup-iou",
        type=float,
        default=0.65,
    )

    parser.add_argument(
        "--detect-every",
        type=int,
        default=1,
        help="每 N 帧运行一次 YOLO",
    )

    parser.add_argument(
        "--match-iou",
        type=float,
        default=0.25,
    )

    parser.add_argument(
        "--max-coast",
        type=int,
        default=75,
    )

    parser.add_argument(
        "--min-hits",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--hand-model",
        default=str(HAND_MODEL_PATH),
        help="MediaPipe hand_landmarker.task",
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

    args = parser.parse_args()

    # --------------------------------------------------------
    # Object Specs
    # --------------------------------------------------------

    specs = build_specs(args)

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
            num_hands=2,
        )

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
                    if distance < 180:

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
            # 信息面板
            # ------------------------------------------------

            draw_info_panel(
                canvas,
                hands,
                result.tracks,
                display_fps,
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