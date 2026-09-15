from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import time

import yaml


@dataclass
class ActiveObjectState:
    track_id: int
    object_id: str
    label: str
    last_center: tuple[float, float]
    last_speed: float = 0.0

    near_since: float | None = None

    # 本次运动开始时间
    moving_since: float | None = None

    # 本次进入低速状态的时间
    stopped_since: float | None = None

    held_since: float | None = None
    placed_since: float | None = None

    last_event_at: dict[str, float] = field(default_factory=dict)
    last_seen: float = 0.0


class ActivityEngine:
    """Generic temporal reasoning engine.

    Vision supplies standardized perception JSON.
    Activity YAML supplies thresholds and semantic completion rules.
    The engine emits sparse semantic events for the Agent.
    """

    def __init__(self, config_path: str | Path, window_seconds: float = 2.0):
        self.config_path = Path(config_path)

        raw = yaml.safe_load(
            self.config_path.read_text(encoding="utf-8")
        ) or {}

        self.config = raw.get("activity", raw)

        self.activity_id = self.config.get(
            "id",
            self.config_path.stem
        )

        self.activity_name = self.config.get(
            "name",
            self.activity_id
        )

        self.window_seconds = float(window_seconds)

        self.history: deque[dict] = deque()

        self.objects: dict[int, ActiveObjectState] = {}

        self.event_counter = 0
        self.last_completed = False
        self.total_event_count = 0
        self.last_state = "IDLE"

        defaults = self.config.get("engine", {})

        self.near_distance = float(
            defaults.get(
                "hand_near_distance_px",
                120.0
            )
        )

        self.moving_speed = float(
            defaults.get(
                "moving_speed_px_s",
                45.0
            )
        )

        self.stopped_speed = float(
            defaults.get(
                "stopped_speed_px_s",
                30.0
            )
        )

        self.persistence_ms = float(
            defaults.get(
                "persistence_ms",
                250.0
            )
        )

        self.event_cooldown_ms = float(
            defaults.get(
                "event_cooldown_ms",
                450.0
            )
        )

        self.stack_vertical_gap_px = float(
            defaults.get(
                "stack_vertical_gap_px",
                45.0
            )
        )

        self.stack_horizontal_ratio = float(
            defaults.get(
                "stack_horizontal_ratio",
                0.65
            )
        )

        self.completion = (
            self.config.get("completion", {}) or {}
        )

    def reset(self) -> None:
        self.history.clear()
        self.objects.clear()

        self.event_counter = 0
        self.last_completed = False
        self.total_event_count = 0
        self.last_state = "IDLE"

    def update(self, perception: dict) -> list[dict]:
        now = float(
            perception.get(
                "timestamp",
                time.time()
            )
        )

        self.history.append(perception)

        while (
            self.history
            and now - float(
                self.history[0].get(
                    "timestamp",
                    now
                )
            ) > self.window_seconds
        ):
            self.history.popleft()

        events: list[dict] = []

        objects = perception.get(
            "objects",
            []
        ) or []

        hands = perception.get(
            "hands",
            []
        ) or []

        relations = perception.get(
            "relations",
            []
        ) or []

        relation_map: dict[
            tuple[str, int],
            float
        ] = {}

        for relation in relations:
            if relation.get("type") != "HAND_NEAR_OBJECT":
                continue

            key = (
                str(
                    relation.get(
                        "hand_id",
                        "Unknown"
                    )
                ),
                int(
                    relation.get(
                        "track_id",
                        -1
                    )
                )
            )

            relation_map[key] = min(
                relation_map.get(
                    key,
                    float("inf")
                ),
                float(
                    relation.get(
                        "distance",
                        9999.0
                    )
                )
            )

        for obj in objects:
            tid = int(
                obj.get(
                    "track_id",
                    -1
                )
            )

            if tid < 0:
                continue

            center = obj.get(
                "center",
                {}
            )

            center_xy = (
                float(
                    center.get(
                        "x",
                        0.0
                    )
                ),
                float(
                    center.get(
                        "y",
                        0.0
                    )
                )
            )

            speed = float(
                obj.get(
                    "motion",
                    {}
                ).get(
                    "speed",
                    0.0
                )
            )

            state = self.objects.get(tid)

            if state is None:
                state = ActiveObjectState(
                    tid,
                    str(
                        obj.get(
                            "object_id",
                            "unknown"
                        )
                    ),
                    str(
                        obj.get(
                            "label",
                            obj.get(
                                "object_id",
                                "物体"
                            )
                        )
                    ),
                    center_xy,
                    last_seen=now
                )

                self.objects[tid] = state

                events.append(
                    self._event(
                        "OBJECT_APPEARED",
                        now,
                        tid,
                        confidence=float(
                            obj.get(
                                "confidence",
                                0.0
                            )
                        )
                    )
                )

            state.last_center = center_xy
            state.last_speed = speed
            state.last_seen = now

            # --------------------------------------------------
            # 1. 手靠近物体
            # --------------------------------------------------
            nearby = any(
                distance <= self.near_distance
                for (
                    hand_id,
                    track_id
                ), distance in relation_map.items()
                if track_id == tid
            )

            if nearby:
                if state.near_since is None:
                    state.near_since = now

                if (
                    self._persisted(
                        state.near_since,
                        now
                    )
                    and self._can_emit(
                        state,
                        "HAND_NEAR_OBJECT",
                        now
                    )
                ):
                    events.append(
                        self._event(
                            "HAND_NEAR_OBJECT",
                            now,
                            tid,
                            confidence=self._distance_confidence(
                                tid,
                                relation_map
                            )
                        )
                    )
            else:
                state.near_since = None

            # --------------------------------------------------
            # 2. 物体运动
            # --------------------------------------------------
            if speed >= self.moving_speed:

                # 如果重新开始运动，开始一个新的运动周期
                if state.moving_since is None:
                    state.moving_since = now

                # 只要重新进入运动状态，停止计时立即失效
                state.stopped_since = None

                if (
                    self._persisted(
                        state.moving_since,
                        now
                    )
                    and self._can_emit(
                        state,
                        "OBJECT_MOVING",
                        now
                    )
                ):
                    events.append(
                        self._event(
                            "OBJECT_MOVING",
                            now,
                            tid,
                            confidence=self._speed_confidence(
                                speed
                            )
                        )
                    )

            # --------------------------------------------------
            # 3. 物体停止
            #
            # 关键修改：
            # 不再要求 stopped_since 必须依赖 moving_since
            # 每次进入低速区间都独立开始计时。
            # --------------------------------------------------
            elif speed <= self.stopped_speed:

                if state.stopped_since is None:
                    state.stopped_since = now

                stopped_duration_ms = (
                    now - state.stopped_since
                ) * 1000.0

                # 只有确实经历过运动，才认为这是“运动后的停止”
                has_moved = (
                    state.moving_since is not None
                )

                if (
                    has_moved
                    and stopped_duration_ms
                    >= self.persistence_ms
                    and self._can_emit(
                        state,
                        "OBJECT_STOPPED",
                        now
                    )
                ):
                    events.append(
                        self._event(
                            "OBJECT_STOPPED",
                            now,
                            tid,
                            confidence=self._stopped_confidence(
                                speed
                            )
                        )
                    )

                    # 一个完整的运动周期已经结束
                    # 清除 moving_since，防止同一次停止重复触发
                    state.moving_since = None

            # --------------------------------------------------
            # 4. PICK_UP
            # --------------------------------------------------
            if nearby and speed >= self.moving_speed:

                if state.held_since is None:
                    state.held_since = now

                if (
                    self._persisted(
                        state.held_since,
                        now
                    )
                    and self._can_emit(
                        state,
                        "PICK_UP",
                        now
                    )
                ):
                    events.append(
                        self._event(
                            "PICK_UP",
                            now,
                            tid,
                            confidence=min(
                                1.0,
                                0.55 + speed / 400.0
                            )
                        )
                    )

            elif not nearby:
                state.held_since = None

            # --------------------------------------------------
            # 5. PLACE
            #
            # STOPPED 之后，如果手离开，则认为放置完成。
            # --------------------------------------------------
            if (
                speed <= self.stopped_speed
                and state.stopped_since is not None
                and state.moving_since is None
                and not nearby
            ):
                if self._can_emit(
                    state,
                    "PLACE",
                    now
                ):
                    events.append(
                        self._event(
                            "PLACE",
                            now,
                            tid,
                            confidence=0.88
                        )

                    )

                    state.placed_since = now

                    # 停止周期结束
                    state.stopped_since = None

        # ------------------------------------------------------
        # 6. 堆叠检测
        # ------------------------------------------------------
        events.extend(
            self._detect_stacks(
                perception,
                now
            )
        )

        # ------------------------------------------------------
        # 7. 活动完成检测
        # ------------------------------------------------------
        completion_event = self._check_completion(
            perception,
            now
        )

        if completion_event is not None:
            events.append(
                completion_event
            )

        if events:
            self.total_event_count += len(events)

        self.last_state = self._derive_state(
            perception
        )

        return events

    def _detect_stacks(
        self,
        perception: dict,
        now: float
    ) -> list[dict]:

        objs = perception.get(
            "objects",
            []
        ) or []

        events = []

        for i, a in enumerate(objs):
            for b in objs[i + 1:]:

                if str(
                    a.get("object_id")
                ) != str(
                    b.get("object_id")
                ):
                    continue

                tid_a = int(
                    a.get(
                        "track_id",
                        -1
                    )
                )

                tid_b = int(
                    b.get(
                        "track_id",
                        -1
                    )
                )

                if tid_a < 0 or tid_b < 0:
                    continue

                bbox_a = self._get_bbox(a)
                bbox_b = self._get_bbox(b)

                if bbox_a is None or bbox_b is None:
                    continue

                ax1, ay1, ax2, ay2 = bbox_a
                bx1, by1, bx2, by2 = bbox_b

                aw = max(
                    1.0,
                    ax2 - ax1
                )

                bw = max(
                    1.0,
                    bx2 - bx1
                )

                ah = max(
                    1.0,
                    ay2 - ay1
                )

                bh = max(
                    1.0,
                    by2 - by1
                )

                horizontal_overlap = max(
                    0.0,
                    min(ax2, bx2)
                    - max(ax1, bx1)
                )

                overlap_ratio = (
                    horizontal_overlap
                    / min(aw, bw)
                )

                vertical_gap = min(
                    abs(ay2 - by1),
                    abs(by2 - ay1)
                )

                if (
                    overlap_ratio
                    >= self.stack_horizontal_ratio
                    and vertical_gap
                    <= self.stack_vertical_gap_px
                ):
                    key_tid = min(
                        tid_a,
                        tid_b
                    )

                    state = self.objects.get(
                        key_tid
                    )

                    if state is None:
                        continue

                    if self._can_emit(
                        state,
                        "STACKED",
                        now
                    ):
                        confidence = min(
                            1.0,
                            0.55
                            + overlap_ratio * 0.4
                        )

                        events.append(
                            self._event(
                                "STACKED",
                                now,
                                key_tid,
                                secondary_track_id=max(
                                    tid_a,
                                    tid_b
                                ),
                                confidence=confidence
                            )
                        )

        return events

    def _check_completion(
        self,
        perception: dict,
        now: float
    ) -> dict | None:

        if self.last_completed:
            return None

        required = int(
            self.completion.get(
                "min_stacked_objects",
                0
            ) or 0
        )

        if required <= 0:
            return None

        objects = perception.get(
            "objects",
            []
        ) or []

        if len(objects) < required:
            return None

        stacked_ids: set[int] = set()

        for sample in self.history:

            sample_objects = sample.get(
                "objects",
                []
            ) or []

            for i, a in enumerate(
                sample_objects
            ):
                for b in sample_objects[i + 1:]:

                    if str(
                        a.get("object_id")
                    ) != str(
                        b.get("object_id")
                    ):
                        continue

                    bbox_a = self._get_bbox(a)
                    bbox_b = self._get_bbox(b)

                    if bbox_a is None or bbox_b is None:
                        continue

                    ax1, ay1, ax2, ay2 = bbox_a
                    bx1, by1, bx2, by2 = bbox_b

                    aw = max(
                        1.0,
                        ax2 - ax1
                    )

                    bw = max(
                        1.0,
                        bx2 - bx1
                    )

                    horizontal_overlap = max(
                        0.0,
                        min(ax2, bx2)
                        - max(ax1, bx1)
                    )

                    overlap_ratio = (
                        horizontal_overlap
                        / min(aw, bw)
                    )

                    vertical_gap = min(
                        abs(ay2 - by1),
                        abs(by2 - ay1)
                    )

                    if (
                        overlap_ratio
                        >= self.stack_horizontal_ratio
                        and vertical_gap
                        <= self.stack_vertical_gap_px
                    ):
                        stacked_ids.add(
                            int(
                                a.get(
                                    "track_id",
                                    -1
                                )
                            )
                        )

                        stacked_ids.add(
                            int(
                                b.get(
                                    "track_id",
                                    -1
                                )
                            )
                        )

        stacked_ids.discard(-1)

        if len(stacked_ids) < required:
            return None

        stable = all(
            float(
                o.get(
                    "motion",
                    {}
                ).get(
                    "speed",
                    0.0
                )
            ) <= self.stopped_speed
            for o in objects
        )

        if stable:
            self.last_completed = True

            confidence = min(
                1.0,
                0.65
                + len(stacked_ids)
                / max(
                    required * 4.0,
                    1.0
                )
            )

            return self._event(
                "ACTIVITY_COMPLETED",
                now,
                None,
                confidence=confidence
            )

        return None

    def _derive_state(
        self,
        perception: dict
    ) -> str:

        if not perception.get(
            "objects"
        ):
            return "IDLE"

        speeds = [
            float(
                o.get(
                    "motion",
                    {}
                ).get(
                    "speed",
                    0.0
                )
            )
            for o in perception.get(
                "objects",
                []
            )
        ]

        relations = perception.get(
            "relations",
            []
        ) or []

        if (
            any(
                r.get("type")
                == "HAND_NEAR_OBJECT"
                for r in relations
            )
            and any(
                s >= self.moving_speed
                for s in speeds
            )
        ):
            return "MOVING_OBJECT"

        if any(
            r.get("type")
            == "HAND_NEAR_OBJECT"
            for r in relations
        ):
            return "HAND_APPROACHING"

        if any(
            s >= self.moving_speed
            for s in speeds
        ):
            return "OBJECT_MOVING"

        return "OBSERVING"

    def _persisted(
        self,
        since: float | None,
        now: float
    ) -> bool:

        return (
            since is not None
            and (
                now - since
            ) * 1000.0
            >= self.persistence_ms
        )

    def _can_emit(
        self,
        state: ActiveObjectState,
        event_type: str,
        now: float
    ) -> bool:

        last = state.last_event_at.get(
            event_type
        )

        if (
            last is not None
            and (
                now - last
            ) * 1000.0
            < self.event_cooldown_ms
        ):
            return False

        state.last_event_at[
            event_type
        ] = now

        return True

    def _distance_confidence(
        self,
        tid: int,
        relation_map: dict[
            tuple[str, int],
            float
        ]
    ) -> float:

        d = min(
            (
                distance
                for (
                    _,
                    track_id
                ), distance
                in relation_map.items()
                if track_id == tid
            ),
            default=self.near_distance
        )

        return max(
            0.0,
            min(
                1.0,
                1.0
                - d
                / max(
                    self.near_distance,
                    1.0
                )
            )
        )

    @staticmethod
    def _speed_confidence(
        speed: float
    ) -> float:

        return max(
            0.0,
            min(
                1.0,
                0.55
                + speed / 500.0
            )
        )

    def _stopped_confidence(
        self,
        speed: float
    ) -> float:

        return max(
            0.0,
            min(
                1.0,
                0.98
                - speed
                / max(
                    self.stopped_speed * 2.0,
                    1.0
                )
            )
        )

    @staticmethod
    def _get_bbox(
        obj: dict
    ) -> tuple[
        float,
        float,
        float,
        float
    ] | None:

        bbox = obj.get("bbox")

        if (
            isinstance(bbox, (list, tuple))
            and len(bbox) >= 4
        ):
            return (
                float(bbox[0]),
                float(bbox[1]),
                float(bbox[2]),
                float(bbox[3])
            )

        return None

    def _event(
        self,
        event_type: str,
        timestamp: float,
        track_id: int | None,
        *,
        confidence: float,
        secondary_track_id: int | None = None
    ) -> dict:

        self.event_counter += 1

        event = {
            "schema_version": "1.0",
            "timestamp": timestamp,
            "activity": {
                "id": self.activity_id,
                "name": self.activity_name
            },
            "event": {
                "id": (
                    f"evt_"
                    f"{self.event_counter:06d}"
                ),
                "type": event_type,
                "confidence": round(
                    float(
                        max(
                            0.0,
                            min(
                                1.0,
                                confidence
                            )
                        )
                    ),
                    4
                ),
                "actor": {
                    "type": "child"
                },
            },
        }

        if track_id is not None:
            event["event"]["object"] = {
                "object_id": "paper_cup",
                "track_id": track_id
            }

        if secondary_track_id is not None:
            event["event"][
                "secondary_object"
            ] = {
                "track_id": secondary_track_id
            }

        return event