from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
import time
from typing import Any

import yaml


@dataclass
class ActiveObjectState:
    """单个被追踪物体在 Activity Engine 中的状态。"""

    track_id: int
    object_id: str
    label: str

    last_center: tuple[float, float]

    last_speed: float = 0.0
    last_seen: float = 0.0

    # 手靠近物体的开始时间
    near_since: float | None = None

    # 当前运动周期开始时间
    moving_since: float | None = None

    # 当前停止周期开始时间
    stopped_since: float | None = None

    # 可能正在拿着物体的开始时间
    held_since: float | None = None

    # 最近一次放置时间
    placed_since: float | None = None

    # 标记物体在本次停止前是否真正经历过运动（用于精准触发 OBJECT_STOPPED 和 PLACE）
    has_moved: bool = False

    # 每种事件最近一次触发时间
    last_event_at: dict[str, float] = field(default_factory=dict)


class ActivityEngine:
    """
    通用 Activity Engine。

    职责：
        1. 接收 Vision Layer 输出的标准化 perception JSON
        2. 结合时间窗口判断基础行为
        3. 输出通用 Activity Events
        4. 输出稳定的通用 Activity State

    不负责：
        - 判断“纸杯是不是叠好了”
        - 判断“衣服是不是叠好了”
        - 判断“画是不是画完了”
        - 判断具体活动语义
        - 判断活动是否完成

    这些高级语义判断交给上层 Activity Understanding / VLM。
    """

    def __init__(
        self,
        config_path: str | Path,
        window_seconds: float = 2.0,
    ):
        self.config_path = Path(config_path)

        raw = yaml.safe_load(
            self.config_path.read_text(encoding="utf-8")
        ) or {}

        self.config = raw.get("activity", raw)

        self.activity_id = self.config.get(
            "id",
            self.config_path.stem,
        )

        self.activity_name = self.config.get(
            "name",
            self.activity_id,
        )

        self.window_seconds = float(window_seconds)

        # --------------------------------------------------
        # 时间历史（设定 maxlen 防止异常情况下的内存膨胀）
        # --------------------------------------------------
        # 假设最大帧率为 60fps，预留双倍安全空间
        max_history_len = int(self.window_seconds * 120)
        self.history: deque[dict[str, Any]] = deque(maxlen=max_history_len)

        # --------------------------------------------------
        # 当前追踪物体
        # --------------------------------------------------

        self.objects: dict[int, ActiveObjectState] = {}

        # --------------------------------------------------
        # Event
        # --------------------------------------------------

        self.event_counter = 0
        self.total_event_count = 0

        # --------------------------------------------------
        # Activity State
        # --------------------------------------------------

        self.last_state = "IDLE"

        # State 防抖
        self.candidate_state: str | None = None
        self.candidate_since: float | None = None

        # State 最短持续时间（支持从 YAML 配置文件中覆盖）
        default_state_persistence = {
            "IDLE": 500.0,
            "OBSERVING": 600.0,
            "HAND_APPROACHING": 300.0,
            "INTERACTING": 300.0,
            "HOLDING_OBJECT": 400.0,
            "MOVING_OBJECT": 300.0,
            "PLACING_OBJECT": 500.0,
        }
        custom_state_persistence = self.config.get("state_persistence_ms", {})
        self.state_persistence_ms = {
            **default_state_persistence,
            **custom_state_persistence,
        }

        # --------------------------------------------------
        # 通用 Engine 参数
        # --------------------------------------------------

        defaults = self.config.get("engine", {}) or {}

        # 手与物体的最大接近距离
        self.near_distance = float(
            defaults.get(
                "hand_near_distance_px",
                120.0,
            )
        )

        # 认为物体正在运动的速度
        self.moving_speed = float(
            defaults.get(
                "moving_speed_px_s",
                45.0,
            )
        )

        # 认为物体已经停止的速度
        self.stopped_speed = float(
            defaults.get(
                "stopped_speed_px_s",
                30.0,
            )
        )

        # 状态 / 事件必须持续的时间
        self.persistence_ms = float(
            defaults.get(
                "persistence_ms",
                250.0,
            )
        )

        # 同一种 Event 两次触发之间的最短间隔
        self.event_cooldown_ms = float(
            defaults.get(
                "event_cooldown_ms",
                450.0,
            )
        )

        # 超时清理过期 Track 的系数
        self.stale_timeout_seconds = float(
            defaults.get(
                "stale_timeout_seconds",
                self.window_seconds * 2.0,
            )
        )

        # 置信度计算相关参数参数化
        self.place_confidence_base = float(defaults.get("place_confidence_base", 0.88))
        self.pickup_speed_scale = float(defaults.get("pickup_speed_scale", 400.0))
        self.moving_speed_scale = float(defaults.get("moving_speed_scale", 500.0))

    # ======================================================
    # Public API
    # ======================================================

    def reset(self) -> None:
        """重置 Activity Engine。"""

        self.history.clear()
        self.objects.clear()

        self.event_counter = 0
        self.total_event_count = 0

        self.last_state = "IDLE"

        self.candidate_state = None
        self.candidate_since = None

    def update(self, perception: dict) -> list[dict]:
        """
        输入一帧 perception，返回这一帧新产生的 Events。
        """

        now = float(
            perception.get(
                "timestamp",
                time.time(),
            )
        )

        # --------------------------------------------------
        # 保存时间窗口
        # --------------------------------------------------

        self.history.append(perception)

        while (
            self.history
            and now
            - float(
                self.history[0].get(
                    "timestamp",
                    now,
                )
            )
            > self.window_seconds
        ):
            self.history.popleft()

        # --------------------------------------------------
        # 获取 Vision 数据
        # --------------------------------------------------

        objects = perception.get(
            "objects",
            [],
        ) or []

        relations = perception.get(
            "relations",
            [],
        ) or []

        # --------------------------------------------------
        # 建立 HAND_NEAR_OBJECT 关系 (优化映射结构，提升性能)
        # key: track_id, value: min_distance
        # --------------------------------------------------

        relation_map: dict[int, float] = {}

        for relation in relations:
            if relation.get("type") != "HAND_NEAR_OBJECT":
                continue

            track_id = int(relation.get("track_id", -1))
            if track_id < 0:
                continue

            distance = float(relation.get("distance", 9999.0))
            relation_map[track_id] = min(
                relation_map.get(track_id, float("inf")),
                distance,
            )

        # ==================================================
        # 处理所有物体
        # ==================================================

        events: list[dict] = []

        for obj in objects:

            track_id = int(
                obj.get(
                    "track_id",
                    -1,
                )
            )

            if track_id < 0:
                continue

            # --------------------------------------------------
            # 基础物体信息
            # --------------------------------------------------

            object_id = str(
                obj.get(
                    "object_id",
                    "unknown",
                )
            )

            label = str(
                obj.get(
                    "label",
                    object_id,
                )
            )

            center = obj.get(
                "center",
                {},
            )

            center_xy = (
                float(
                    center.get(
                        "x",
                        0.0,
                    )
                ),
                float(
                    center.get(
                        "y",
                        0.0,
                    )
                ),
            )

            motion = obj.get(
                "motion",
                {},
            ) or {}

            speed = float(
                motion.get(
                    "speed",
                    0.0,
                )
            )

            # --------------------------------------------------
            # 获取 / 创建物体状态
            # --------------------------------------------------

            state = self.objects.get(track_id)

            if state is None:

                state = ActiveObjectState(
                    track_id=track_id,
                    object_id=object_id,
                    label=label,
                    last_center=center_xy,
                    last_speed=speed,
                    last_seen=now,
                )

                self.objects[track_id] = state

                events.append(
                    self._event(
                        event_type="OBJECT_APPEARED",
                        timestamp=now,
                        track_id=track_id,
                        object_id=object_id,
                        label=label,
                        confidence=float(
                            obj.get(
                                "confidence",
                                0.0,
                            )
                        ),
                    )
                )

            # 更新状态
            state.object_id = object_id
            state.label = label
            state.last_center = center_xy
            state.last_speed = speed
            state.last_seen = now

            # --------------------------------------------------
            # 判断手是否靠近
            # --------------------------------------------------

            hand_distance = relation_map.get(track_id)
            nearby = hand_distance is not None and hand_distance <= self.near_distance

            # ==================================================
            # 1. HAND_NEAR_OBJECT
            # ==================================================

            if nearby:

                if state.near_since is None:
                    state.near_since = now

                if (
                    self._persisted(
                        state.near_since,
                        now,
                    )
                    and self._can_emit(
                        state,
                        "HAND_NEAR_OBJECT",
                        now,
                    )
                ):

                    events.append(
                        self._event(
                            event_type="HAND_NEAR_OBJECT",
                            timestamp=now,
                            track_id=track_id,
                            object_id=object_id,
                            label=label,
                            confidence=self._distance_confidence(
                                hand_distance
                            ),
                        )
                    )

            else:

                state.near_since = None

            # ==================================================
            # 2. OBJECT_MOVING
            # ==================================================

            if speed >= self.moving_speed:

                if state.moving_since is None:
                    state.moving_since = now

                # 标记该物体经历了有效的运动过程
                state.has_moved = True

                # 重新运动后，当前停止周期失效
                state.stopped_since = None

                if (
                    self._persisted(
                        state.moving_since,
                        now,
                    )
                    and self._can_emit(
                        state,
                        "OBJECT_MOVING",
                        now,
                    )
                ):

                    events.append(
                        self._event(
                            event_type="OBJECT_MOVING",
                            timestamp=now,
                            track_id=track_id,
                            object_id=object_id,
                            label=label,
                            confidence=self._speed_confidence(
                                speed
                            ),
                        )
                    )

            # ==================================================
            # 3. OBJECT_STOPPED
            # ==================================================

            elif speed <= self.stopped_speed:

                if state.stopped_since is None:
                    state.stopped_since = now

                stopped_duration_ms = (
                    now - state.stopped_since
                ) * 1000.0

                # 必须之前经历过运动才认为这是有效的“停止”
                if (
                    state.has_moved
                    and stopped_duration_ms >= self.persistence_ms
                    and self._can_emit(
                        state,
                        "OBJECT_STOPPED",
                        now,
                    )
                ):

                    events.append(
                        self._event(
                            event_type="OBJECT_STOPPED",
                            timestamp=now,
                            track_id=track_id,
                            object_id=object_id,
                            label=label,
                            confidence=self._stopped_confidence(
                                speed
                            ),
                        )
                    )

            # ==================================================
            # 4. PICK_UP
            #
            # 手靠近 + 物体持续运动
            # ==================================================

            if nearby and speed >= self.moving_speed:

                if state.held_since is None:
                    state.held_since = now

                if (
                    self._persisted(
                        state.held_since,
                        now,
                    )
                    and self._can_emit(
                        state,
                        "PICK_UP",
                        now,
                    )
                ):

                    events.append(
                        self._event(
                            event_type="PICK_UP",
                            timestamp=now,
                            track_id=track_id,
                            object_id=object_id,
                            label=label,
                            confidence=min(
                                1.0,
                                0.55 + speed / self.pickup_speed_scale,
                            ),
                        )
                    )

            elif not nearby:

                state.held_since = None

            # ==================================================
            # 5. PLACE
            #
            # 物体经历过运动 + 当前已经停止 + 手离开
            # ==================================================

            if (
                speed <= self.stopped_speed
                and state.stopped_since is not None
                and state.has_moved
                and not nearby
            ):

                if self._can_emit(
                    state,
                    "PLACE",
                    now,
                ):

                    events.append(
                        self._event(
                            event_type="PLACE",
                            timestamp=now,
                            track_id=track_id,
                            object_id=object_id,
                            label=label,
                            confidence=self.place_confidence_base,
                        )
                    )

                    state.placed_since = now

                    # 一次放置行为完成后，重置状态标记
                    state.stopped_since = None
                    state.moving_since = None
                    state.has_moved = False

        # ==================================================
        # 清理长时间没有出现的 Track
        # ==================================================

        self._cleanup_stale_objects(
            now=now,
        )

        # ==================================================
        # Event 统计
        # ==================================================

        if events:
            self.total_event_count += len(events)

        # ==================================================
        # Activity State
        # ==================================================

        raw_state = self._derive_raw_state(
            perception=perception,
        )

        self.last_state = self._stabilize_state(
            raw_state=raw_state,
            now=now,
        )

        return events

    # ======================================================
    # Activity State
    # ======================================================

    def _derive_raw_state(
        self,
        perception: dict,
    ) -> str:
        """
        根据当前 perception 判断“当前正在发生什么”。

        注意：
        这里只判断通用行为，不判断具体活动。
        """

        objects = perception.get(
            "objects",
            [],
        ) or []

        relations = perception.get(
            "relations",
            [],
        ) or []

        # 没有检测到物体
        if not objects:
            return "IDLE"

        # ----------------------------------------------
        # 当前是否存在手-物体关系
        # ----------------------------------------------

        hand_near = any(
            relation.get("type") == "HAND_NEAR_OBJECT"
            for relation in relations
        )

        # ----------------------------------------------
        # 当前是否有高速运动物体
        # ----------------------------------------------

        moving = any(
            float(
                obj.get(
                    "motion",
                    {},
                ).get(
                    "speed",
                    0.0,
                )
            )
            >= self.moving_speed
            for obj in objects
        )

        # ----------------------------------------------
        # 当前是否有刚刚停止的物体
        # ----------------------------------------------

        stopped = any(
            float(
                obj.get(
                    "motion",
                    {},
                ).get(
                    "speed",
                    0.0,
                )
            )
            <= self.stopped_speed
            for obj in objects
        )

        # ----------------------------------------------
        # 状态判断
        # ----------------------------------------------

        if hand_near and moving:
            return "MOVING_OBJECT"

        if hand_near:

            # 如果手靠近物体，但物体没有明显运动，
            # 表示正在发生交互。
            return "INTERACTING"

        if moving:
            return "MOVING_OBJECT"

        if stopped:
            return "OBSERVING"

        return "OBSERVING"

    def _stabilize_state(
        self,
        raw_state: str,
        now: float,
    ) -> str:
        """
        Activity State 防抖。

        不因为单帧变化就立即切换状态。
        """

        # 当前状态与 raw state 一样
        if raw_state == self.last_state:

            self.candidate_state = None
            self.candidate_since = None

            return self.last_state

        # 新候选状态
        if self.candidate_state != raw_state:

            self.candidate_state = raw_state
            self.candidate_since = now

            # 第一次出现时保持原状态
            return self.last_state

        # 候选状态持续时间
        if self.candidate_since is None:
            self.candidate_since = now
            return self.last_state

        duration_ms = (
            now - self.candidate_since
        ) * 1000.0

        required_ms = self.state_persistence_ms.get(
            raw_state,
            400.0,
        )

        if duration_ms >= required_ms:

            self.last_state = raw_state

            self.candidate_state = None
            self.candidate_since = None

        return self.last_state

    # ======================================================
    # Event 工具
    # ======================================================

    def _persisted(
        self,
        since: float | None,
        now: float,
    ) -> bool:
        """判断某个状态是否持续足够长时间。"""

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
        now: float,
    ) -> bool:
        """
        Event cooldown。

        防止：
            OBJECT_MOVING
            OBJECT_MOVING
            ...
        每一帧疯狂输出。
        """

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

        state.last_event_at[event_type] = now

        return True

    # ======================================================
    # Confidence
    # ======================================================

    def _distance_confidence(
        self,
        distance: float | None,
    ) -> float:
        if distance is None:
            distance = self.near_distance

        return max(
            0.0,
            min(
                1.0,
                1.0
                - distance
                / max(
                    self.near_distance,
                    1.0,
                ),
            ),
        )

    def _speed_confidence(
        self,
        speed: float,
    ) -> float:

        return max(
            0.0,
            min(
                1.0,
                0.55
                + speed / self.moving_speed_scale,
            ),
        )

    def _stopped_confidence(
        self,
        speed: float,
    ) -> float:

        return max(
            0.0,
            min(
                1.0,
                0.98
                - speed
                / max(
                    self.stopped_speed * 2.0,
                    1.0,
                ),
            ),
        )

    # ======================================================
    # Object Cleanup
    # ======================================================

    def _cleanup_stale_objects(
        self,
        now: float,
    ) -> None:
        """
        清理已经长时间没有出现在 perception 中的物体。

        当前只清理内部状态，不产生
        OBJECT_DISAPPEARED Event。
        """

        stale_ids = [
            track_id
            for track_id, state in self.objects.items()
            if now - state.last_seen > self.stale_timeout_seconds
        ]

        for track_id in stale_ids:
            self.objects.pop(
                track_id,
                None,
            )

    # ======================================================
    # BBox 工具
    # ======================================================

    @staticmethod
    def _get_bbox(
        obj: dict,
    ) -> tuple[
        float,
        float,
        float,
        float,
    ] | None:

        bbox = obj.get("bbox")

        if (
            isinstance(
                bbox,
                (list, tuple),
            )
            and len(bbox) >= 4
        ):

            return (
                float(bbox[0]),
                float(bbox[1]),
                float(bbox[2]),
                float(bbox[3]),
            )

        return None

    # ======================================================
    # Event 构造
    # ======================================================

    def _event(
        self,
        event_type: str,
        timestamp: float,
        track_id: int | None,
        *,
        object_id: str | None = None,
        label: str | None = None,
        confidence: float,
    ) -> dict:

        self.event_counter += 1

        event = {
            "schema_version": "1.0",
            "timestamp": timestamp,
            "activity": {
                "id": self.activity_id,
                "name": self.activity_name,
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
                                confidence,
                            ),
                        )
                    ),
                    4,
                ),
                "actor": {
                    "type": "child",
                },
            },
        }

        if track_id is not None:

            object_data = {
                "track_id": track_id,
            }

            if object_id is not None:
                object_data["object_id"] = object_id

            if label is not None:
                object_data["label"] = label

            event["event"]["object"] = object_data

        return event