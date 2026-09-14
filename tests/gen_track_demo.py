#!/usr/bin/env python
"""生成追踪测试视频：真实纸杯贴片 + 运动 + 遮挡。

做法：
  从 test_images/cup_blue_table.jpg 里按检测结果 JSON 的 bbox
  抠出真实纸杯贴片（保证 YOLO-World 真的能检出来），
  在合成桌面上动画出 6 秒场景：
    杯A：3s 起从左侧匀速右移 2.5s（模拟"拿起→搬运→放置"）
    杯B：静止在桌面右侧
    杯C：静止在桌面上
    遮挡物：3.5s~5.5s 盖住杯A（模拟手完全挡住杯子的时刻），
            随杯A移动，杯子被完全盖住 → 检测丢失 → 追踪滑行接管

用法：python tests/gen_track_demo.py
输出：out/track_demo.mp4
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SRC_IMG = ROOT / "test_images" / "cup_blue_table.jpg"
SRC_JSON = ROOT / "out" / "cup_blue_table_detected.json"
OUT = ROOT / "out" / "track_demo.mp4"

W, H = 1280, 720
FPS = 30
SECONDS = 6.0
N_FRAMES = int(FPS * SECONDS)

# 桌面区域与背景
BG_TOP = np.array([150, 90, 40])     # 深蓝灰（上方墙面）
BG_BOTTOM = np.array([200, 120, 60])  # 偏蓝（桌面）


def load_cup_patches(max_n: int = 3) -> list[np.ndarray]:
    data = json.loads(SRC_JSON.read_text(encoding="utf-8"))
    cups = [o for o in data["objects"] if o["object_id"] == "paper_cup"]
    cups.sort(key=lambda o: -o["confidence"])
    img = cv2.imdecode(np.fromfile(str(SRC_IMG), dtype=np.uint8),
                       cv2.IMREAD_COLOR)
    patches = []
    for c in cups[:max_n]:
        x1, y1, x2, y2 = c["bbox_xyxy"]
        pad = int(0.04 * max(x2 - x1, y2 - y1))
        x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
        x2, y2 = min(img.shape[1], x2 + pad), min(img.shape[0], y2 + pad)
        patches.append(img[y1:y2, x1:x2].copy())
    return patches


def scale_patch(patch: np.ndarray, target_h: int) -> np.ndarray:
    h, w = patch.shape[:2]
    s = target_h / h
    return cv2.resize(patch, (max(1, int(w * s)), target_h),
                      interpolation=cv2.INTER_AREA)


def paste(canvas: np.ndarray, patch: np.ndarray, cx: float, bottom: float) -> None:
    """把贴片按中心 x、底边 y 贴到画布上。"""
    ph, pw = patch.shape[:2]
    x1 = int(cx - pw / 2)
    y1 = int(bottom - ph)
    x2, y2 = x1 + pw, y1 + ph
    cx1, cy1 = max(0, x1), max(0, y1)
    cx2, cy2 = min(W, x2), min(H, y2)
    if cx2 <= cx1 or cy2 <= cy1:
        return
    canvas[cy1:cy2, cx1:cx2] = patch[cy1 - y1:cy2 - y1, cx1 - x1:cx2 - x1]


def make_occluder(size: int) -> np.ndarray:
    """肤色圆块模拟完全挡住杯子的手。"""
    img = np.zeros((size, size, 3), dtype=np.uint8)
    cv2.circle(img, (size // 2, size // 2), size // 2 - 2,
               (140, 170, 220), -1)
    cv2.circle(img, (size // 2, size // 2), size // 2 - 2,
               (110, 140, 190), 3)
    return img


def main() -> None:
    if not SRC_JSON.exists() or not SRC_IMG.exists():
        sys.exit(f"缺少素材: {SRC_IMG if not SRC_IMG.exists() else SRC_JSON}")
    patches = load_cup_patches(3)
    if len(patches) < 3:
        sys.exit(f"只能抠到 {len(patches)} 个纸杯贴片（需要 3 个）")

    # 统一目标高度（近大远小：A 最近，C 稍远）
    pA = scale_patch(patches[0], 220)
    pB = scale_patch(patches[1], 190)
    pC = scale_patch(patches[2], 170)
    occ = make_occluder(260)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    vw = cv2.VideoWriter(str(OUT), fourcc, FPS, (W, H))

    # 杯A 运动参数：3.0s~5.5s 从 x=300 移到 x=900，底边 y 不变
    tA0, tA1 = 3.0, 5.5
    xA0, xA1 = 300.0, 900.0
    yBottomA = 640.0
    # 遮挡窗口：3.5s~5.5s 完全盖住杯A
    tO0, tO1 = 3.5, 5.5

    for f in range(N_FRAMES):
        t = f / FPS
        canvas = np.zeros((H, W, 3), dtype=np.float32)
        for y in range(H):  # 竖向渐变：上暗下亮
            a = y / H
            canvas[y, :] = BG_TOP * (1 - a) + BG_BOTTOM * a
        canvas = canvas.astype(np.uint8)
        # 桌面前沿横线
        cv2.line(canvas, (0, 560), (W, 560), (160, 90, 35), 2)

        # 静止杯 B、C
        paste(canvas, pB, 1050, 600)
        paste(canvas, pC, 550, 585)

        # 移动杯 A
        if t < tA0:
            xA = xA0
        elif t >= tA1:
            xA = xA1
        else:
            k = (t - tA0) / (tA1 - tA0)
            xA = xA0 + (xA1 - xA0) * k
        paste(canvas, pA, xA, yBottomA)

        # 遮挡物：盖在杯A上（偏上盖住杯身，底边留一点桌影）
        if tO0 <= t <= tO1:
            paste(canvas, occ, xA, yBottomA + 5)

        # 时间码
        cv2.putText(canvas, f"t={t:.1f}s frame={f}", (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (40, 40, 40), 2,
                    cv2.LINE_AA)
        vw.write(canvas)

    vw.release()
    print(f"已生成: {OUT}  ({N_FRAMES} 帧 @ {FPS}fps, {SECONDS:.0f}s)")
    print("剧情：杯A 于 3.0~5.5s 右移；3.5~5.5s 被肤色圆块完全遮挡（60 帧无检测）")


if __name__ == "__main__":
    main()
