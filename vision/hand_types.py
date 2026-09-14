from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class HandObservation:
    """
    统一的手部感知结果。

    坐标全部使用图像像素坐标：
        x: 0 ~ image_width
        y: 0 ~ image_height

    hand_id:
        "Left" / "Right"

    wrist:
        手腕像素坐标 (x, y)

    bbox:
        手部包围框 [x1, y1, x2, y2]

    landmarks:
        21 个 MediaPipe 手部关键点。
        每个点为 [x, y, z]，其中 x/y 已经转换成像素坐标。
    """

    hand_id: str

    wrist: tuple[float, float]

    bbox: list[float]

    landmarks: list[list[float]] = field(default_factory=list)

    confidence: float = 0.0

    # 原始 handedness score
    handedness_score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "hand_id": self.hand_id,
            "wrist": [
                round(self.wrist[0], 2),
                round(self.wrist[1], 2),
            ],
            "bbox": [
                round(v, 2)
                for v in self.bbox
            ],
            "landmarks": [
                [
                    round(point[0], 4),
                    round(point[1], 4),
                    round(point[2], 4),
                ]
                for point in self.landmarks
            ],
            "confidence": round(self.confidence, 4),
            "handedness_score": round(
                self.handedness_score,
                4,
            ),
        }