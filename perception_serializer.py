from __future__ import annotations

from typing import Any
import time


def _round(value: Any, digits: int = 3):
    try:
        return round(float(value), digits)
    except (TypeError, ValueError):
        return value


def _bbox(value) -> list[float]:
    if value is None:
        return [0.0, 0.0, 0.0, 0.0]
    return [_round(v, 2) for v in list(value)[:4]]


def _point(value) -> dict[str, float]:
    return {"x": _round(value[0], 2), "y": _round(value[1], 2)}


def point_to_bbox_distance(point: tuple[float, float], bbox: list[float]) -> float:
    px, py = point
    x1, y1, x2, y2 = bbox
    dx = max(x1 - px, 0.0, px - x2)
    dy = max(y1 - py, 0.0, py - y2)
    return (dx * dx + dy * dy) ** 0.5


def serialize_perception(
    *,
    frame_index: int,
    timestamp: float | None,
    frame_width: int,
    frame_height: int,
    fps: float,
    hands: list,
    tracker_result,
    activity_id: str,
    activity_name: str,
) -> dict:
    """Convert the current vision output into the stable Activity Engine input schema."""
    ts = time.time() if timestamp is None else float(timestamp)

    hand_items = []
    for hand in hands or []:
        landmarks = []
        for p in getattr(hand, "landmarks", []) or []:
            if len(p) >= 2:
                landmarks.append([_round(p[0], 2), _round(p[1], 2)])

        bbox = _bbox(getattr(hand, "bbox", None))
        wrist_raw = getattr(hand, "wrist", (0.0, 0.0))
        wrist = _point(wrist_raw)
        hand_items.append({
            "hand_id": str(getattr(hand, "hand_id", "Unknown")),
            "confidence": _round(getattr(hand, "confidence", 0.0), 4),
            "bbox": bbox,
            "wrist": wrist,
            "landmarks": landmarks,
        })

    object_items = []
    tracks = getattr(tracker_result, "tracks", []) or []
    for track in tracks:
        if not getattr(track, "confirmed", False):
            continue
        bbox = _bbox(getattr(track, "bbox", None))
        center_raw = getattr(track, "center", (0.0, 0.0))
        width = _round(getattr(track, "width", 0.0), 2)
        height = _round(getattr(track, "height", 0.0), 2)
        vx = _round(getattr(track, "vx", 0.0), 2)
        vy = _round(getattr(track, "vy", 0.0), 2)
        speed = _round(getattr(track, "speed_px_s", 0.0), 2)
        object_items.append({
            "object_id": str(getattr(track, "object_id", "unknown")),
            "track_id": int(getattr(track, "track_id", -1)),
            "label": str(getattr(track, "name_cn", getattr(track, "object_id", "unknown"))),
            "confidence": _round(getattr(track, "confidence", 0.0), 4),
            "bbox": bbox,
            "center": _point(center_raw),
            "size": {"width": width, "height": height},
            "motion": {"vx": vx, "vy": vy, "speed": speed},
            "tracking": {
                "confirmed": bool(getattr(track, "confirmed", False)),
                "predicted": bool(getattr(track, "is_predicted", False)),
                "coasting_frames": int(getattr(track, "coasting_frames", 0)),
                "hits": int(getattr(track, "hits", 0)),
            },
        })

    relations = []
    for hand in hand_items:
        wrist = (hand["wrist"]["x"], hand["wrist"]["y"])
        for obj in object_items:
            distance = point_to_bbox_distance(wrist, obj["bbox"])
            # A generic relation only. Activity-specific thresholds belong to YAML.
            if distance <= 180.0:
                confidence = max(0.0, min(1.0, 1.0 - distance / 180.0))
                relations.append({
                    "type": "HAND_NEAR_OBJECT",
                    "hand_id": hand["hand_id"],
                    "track_id": obj["track_id"],
                    "distance": _round(distance, 2),
                    "confidence": _round(confidence, 4),
                })

    return {
        "schema_version": "1.0",
        "timestamp": ts,
        "frame": {
            "index": int(frame_index),
            "width": int(frame_width),
            "height": int(frame_height),
            "fps": _round(fps, 2),
        },
        "activity": {
            "id": activity_id,
            "name": activity_name,
        },
        "hands": hand_items,
        "objects": object_items,
        "relations": relations,
    }
