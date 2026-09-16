from collections import deque
import math
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

    # 最近一次感知到的速度
    last_speed: float = 0.0

    # 最近一次检测时间
    last_seen: float = 0.0

    # 最近一次物体检测 confidence
    last_object_confidence: float = 0.0

    # --------------------------------------------------
    # 手靠近状态
    # --------------------------------------------------

    near_since: float | None = None

    # 最近一次手物距离
    last_hand_distance: float | None = None

    # --------------------------------------------------
    # 运动状态
    # --------------------------------------------------

    moving_since: float | None = None

    # 停止状态开始时间
    stopped_since: float | None = None

    # --------------------------------------------------
    # PICK_UP 状态
    # --------------------------------------------------

    # 可能正在拿着物体的开始时间
    held_since: float | None = None

    # PICK_UP 是否已经成功触发
    pickup_emitted: bool = False

    # 最近一次 PICK_UP confidence
    pickup_confidence: float = 0.0

    # --------------------------------------------------
    # PLACE 状态
    # --------------------------------------------------

    # 最近一次放置时间
    placed_since: float | None = None

    # --------------------------------------------------
    # 当前动作周期
    # --------------------------------------------------

    # 本次周期中是否经历过有效运动
    has_moved: bool = False

    # 本次周期中是否已经触发 OBJECT_MOVING
    moving_emitted: bool = False

    # 本次周期中是否已经触发 OBJECT_STOPPED
    stopped_emitted: bool = False

    # --------------------------------------------------
    # Event cooldown
    # --------------------------------------------------

    last_event_at: dict[str, float] = field(default_factory=dict)


class ActivityEngine:
    """
    通用 Activity Engine。

    职责：
        1. 接收 Vision Layer 输出的标准化 perception JSON
        2. 根据时间窗口判断基础行为
        3. 输出通用 Activity Events
        4. 输出稳定的通用 Activity State

    不负责：
        - 判断“纸杯是不是叠好了”
        - 判断“衣服是不是叠好了”
        - 判断“画是不是画完了”
        - 判断具体活动语义
        - 判断活动是否完成

    高级活动语义交给 Activity Understanding / VLM。
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

        # ==================================================
        # 时间历史
        # ==================================================

        max_history_len = int(self.window_seconds * 120)

        self.history: deque[dict[str, Any]] = deque(
            maxlen=max_history_len
        )

        # ==================================================
        # 当前追踪物体
        # ==================================================

        self.objects: dict[int, ActiveObjectState] = {}

        # ==================================================
        # Event
        # ==================================================

        self.event_counter = 0
        self.total_event_count = 0

        # ==================================================
        # Activity State
        # ==================================================

        self.last_state = "IDLE"

        self.candidate_state: str | None = None
        self.candidate_since: float | None = None

        # ==================================================
        # State Persistence
        # ==================================================

        default_state_persistence = {
            "IDLE": 500.0,
            "HAND_APPROACHING": 300.0,
            "INTERACTING": 300.0,
            "HOLDING_OBJECT": 400.0,
            "MOVING_OBJECT": 300.0,
            "PLACING_OBJECT": 500.0,
        }

        custom_state_persistence = self.config.get(
            "state_persistence_ms",
            {},
        )

        self.state_persistence_ms = {
            **default_state_persistence,
            **custom_state_persistence,
        }

        # ==================================================
        # 通用 Engine 参数
        # ==================================================

        defaults = self.config.get(
            "engine",
            {},
        ) or {}

        # --------------------------------------------------
        # Hand / Object
        # --------------------------------------------------

        self.near_distance = float(
            defaults.get(
                "hand_near_distance_px",
                120.0,
            )
        )

        # --------------------------------------------------
        # Motion
        # --------------------------------------------------

        self.moving_speed = float(
            defaults.get(
                "moving_speed_px_s",
                45.0,
            )
        )

        self.stopped_speed = float(
            defaults.get(
                "stopped_speed_px_s",
                30.0,
            )
        )

        # --------------------------------------------------
        # Persistence
        # --------------------------------------------------

        self.persistence_ms = float(
            defaults.get(
                "persistence_ms",
                250.0,
            )
        )

        # --------------------------------------------------
        # Event cooldown
        # --------------------------------------------------

        self.event_cooldown_ms = float(
            defaults.get(
                "event_cooldown_ms",
                450.0,
            )
        )

        # --------------------------------------------------
        # Stale timeout
        # --------------------------------------------------

        self.stale_timeout_seconds = float(
            defaults.get(
                "stale_timeout_seconds",
                self.window_seconds * 2.0,
            )
        )

        # ==================================================
        # Confidence 配置
        # ==================================================

        confidence_config = defaults.get(
            "confidence",
            {},
        ) or {}

        # --------------------------------------------------
        # HAND_NEAR_OBJECT
        # --------------------------------------------------

        hand_near_config = confidence_config.get(
            "hand_near",
            {},
        ) or {}

        self.hand_near_conf_min = float(
            hand_near_config.get(
                "min",
                0.60,
            )
        )

        self.hand_near_conf_max = float(
            hand_near_config.get(
                "max",
                0.98,
            )
        )

        # --------------------------------------------------
        # OBJECT_MOVING
        # --------------------------------------------------

        moving_config = confidence_config.get(
            "moving",
            {},
        ) or {}

        self.moving_conf_min = float(
            moving_config.get(
                "min",
                0.60,
            )
        )

        self.moving_conf_max = float(
            moving_config.get(
                "max",
                0.95,
            )
        )

        self.moving_strong_speed = float(
            moving_config.get(
                "strong_speed_px_s",
                max(
                    self.moving_speed * 4.0,
                    180.0,
                ),
            )
        )

        # --------------------------------------------------
        # OBJECT_STOPPED
        # --------------------------------------------------

        stopped_config = confidence_config.get(
            "stopped",
            {},
        ) or {}

        self.stopped_conf_min = float(
            stopped_config.get(
                "min",
                0.65,
            )
        )

        self.stopped_conf_max = float(
            stopped_config.get(
                "max",
                0.98,
            )
        )

        # --------------------------------------------------
        # PICK_UP
        # --------------------------------------------------

        pickup_config = confidence_config.get(
            "pickup",
            {},
        ) or {}

        self.pickup_hand_weight = float(
            pickup_config.get(
                "hand_weight",
                0.35,
            )
        )

        self.pickup_moving_weight = float(
            pickup_config.get(
                "moving_weight",
                0.40,
            )
        )

        self.pickup_object_weight = float(
            pickup_config.get(
                "object_weight",
                0.25,
            )
        )

        self.pickup_conf_min = float(
            pickup_config.get(
                "min",
                0.65,
            )
        )

        self.pickup_conf_max = float(
            pickup_config.get(
                "max",
                0.95,
            )
        )

        # --------------------------------------------------
        # PLACE
        # --------------------------------------------------

        place_config = confidence_config.get(
            "place",
            {},
        ) or {}

        self.place_pickup_weight = float(
            place_config.get(
                "pickup_weight",
                0.25,
            )
        )

        self.place_moving_weight = float(
            place_config.get(
                "moving_weight",
                0.20,
            )
        )

        self.place_stopped_weight = float(
            place_config.get(
                "stopped_weight",
                0.30,
            )
        )

        self.place_hand_away_weight = float(
            place_config.get(
                "hand_away_weight",
                0.25,
            )
        )

        self.place_conf_min = float(
            place_config.get(
                "min",
                0.70,
            )
        )

        self.place_conf_max = float(
            place_config.get(
                "max",
                0.95,
            )
        )

        # ==================================================
        # 兼容旧配置
        # ==================================================

        # 如果 YAML 中没有新的 confidence 配置，
        # 保留旧参数的兼容性。
        self.place_confidence_base = float(
            defaults.get(
                "place_confidence_base",
                0.88,
            )
        )

        self.pickup_speed_scale = float(
            defaults.get(
                "pickup_speed_scale",
                400.0,
            )
        )

        self.moving_speed_scale = float(
            defaults.get(
                "moving_speed_scale",
                500.0,
            )
        )

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

        # ==================================================
        # 保存时间窗口
        # ==================================================

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

        # ==================================================
        # Vision 数据
        # ==================================================

        objects = perception.get(
            "objects",
            [],
        ) or []

        relations = perception.get(
            "relations",
            [],
        ) or []

        # ==================================================
        # 建立手-物体距离映射
        #
        # key:
        #     track_id
        #
        # value:
        #     当前最近手距离
        # ==================================================

        relation_map: dict[int, float] = {}

        for relation in relations:
            relation_type = relation.get("type")

            if relation_type not in {
                "HAND_OBJECT_DISTANCE",
                "HAND_NEAR_OBJECT",
            }:
                continue

            try:
                track_id = int(
                    relation.get(
                        "track_id",
                        -1,
                    )
                )
            except (TypeError, ValueError):
                continue

            if track_id < 0:
                continue

            try:
                distance = float(
                    relation.get(
                        "distance",
                        9999.0,
                    )
                )
            except (TypeError, ValueError):
                continue

            if not math.isfinite(distance):
                continue

            relation_map[track_id] = min(
                relation_map.get(
                    track_id,
                    float("inf"),
                ),
                distance,
            )

        # ==================================================
        # 处理物体
        # ==================================================

        events: list[dict] = []

        for obj in objects:

            try:
                track_id = int(
                    obj.get(
                        "track_id",
                        -1,
                    )
                )
            except (TypeError, ValueError):
                continue

            if track_id < 0:
                continue

            # --------------------------------------------------
            # 基础信息
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
            ) or {}

            try:
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
            except (TypeError, ValueError):
                center_xy = (0.0, 0.0)

            # --------------------------------------------------
            # Motion
            # --------------------------------------------------

            motion = obj.get(
                "motion",
                {},
            ) or {}

            try:
                speed = float(
                    motion.get(
                        "speed",
                        0.0,
                    )
                )
            except (TypeError, ValueError):
                speed = 0.0

            if not math.isfinite(speed):
                speed = 0.0

            # --------------------------------------------------
            # Object confidence
            # --------------------------------------------------

            try:
                object_confidence = float(
                    obj.get(
                        "confidence",
                        0.0,
                    )
                )
            except (TypeError, ValueError):
                object_confidence = 0.0

            object_confidence = self._clamp(
                object_confidence
            )

            # ==================================================
            # 获取 / 创建 Track State
            # ==================================================

            state = self.objects.get(track_id)

            if state is None:

                state = ActiveObjectState(
                    track_id=track_id,
                    object_id=object_id,
                    label=label,
                    last_center=center_xy,
                    last_speed=speed,
                    last_seen=now,
                    last_object_confidence=object_confidence,
                )

                self.objects[track_id] = state

                # ----------------------------------------------
                # OBJECT_APPEARED
                # ----------------------------------------------

                events.append(
                    self._event(
                        event_type="OBJECT_APPEARED",
                        timestamp=now,
                        track_id=track_id,
                        object_id=object_id,
                        label=label,
                        confidence=object_confidence,
                    )
                )

            # ==================================================
            # 更新状态
            # ==================================================

            state.object_id = object_id
            state.label = label
            state.last_center = center_xy
            state.last_speed = speed
            state.last_seen = now
            state.last_object_confidence = object_confidence

            # ==================================================
            # Hand Distance
            # ==================================================

            hand_distance = relation_map.get(track_id)

            state.last_hand_distance = hand_distance

            nearby = (
                hand_distance is not None
                and hand_distance <= self.near_distance
            )

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
                    and not self._event_already_active(
                        state,
                        "HAND_NEAR_OBJECT",
                    )
                    and self._can_emit(
                        state,
                        "HAND_NEAR_OBJECT",
                        now,
                    )
                ):

                    confidence = self._distance_confidence(
                        hand_distance
                    )

                    events.append(
                        self._event(
                            event_type="HAND_NEAR_OBJECT",
                            timestamp=now,
                            track_id=track_id,
                            object_id=object_id,
                            label=label,
                            confidence=confidence,
                        )
                    )

            else:

                # 手离开以后，允许下一次靠近重新触发
                if state.near_since is not None:

                    state.near_since = None

                    # held_since 也结束
                    state.held_since = None

            # ==================================================
            # 2. OBJECT_MOVING
            # ==================================================

            if speed >= self.moving_speed:

                if state.moving_since is None:
                    state.moving_since = now

                # 有效运动
                state.has_moved = True

                # 重新运动以后，停止状态失效
                state.stopped_since = None

                if (
                    self._persisted(
                        state.moving_since,
                        now,
                    )
                    and not state.moving_emitted
                    and self._can_emit(
                        state,
                        "OBJECT_MOVING",
                        now,
                    )
                ):

                    confidence = self._moving_confidence(
                        speed
                    )

                    events.append(
                        self._event(
                            event_type="OBJECT_MOVING",
                            timestamp=now,
                            track_id=track_id,
                            object_id=object_id,
                            label=label,
                            confidence=confidence,
                        )
                    )

                    state.moving_emitted = True

            else:

                # 速度重新回到运动阈值以下，
                # 当前运动阶段结束。
                state.moving_since = None

            # ==================================================
            # 3. OBJECT_STOPPED
            # ==================================================

            if speed <= self.stopped_speed:

                if state.stopped_since is None:
                    state.stopped_since = now

                stopped_duration_ms = (
                    now - state.stopped_since
                ) * 1000.0

                if (
                    state.has_moved
                    and stopped_duration_ms
                    >= self.persistence_ms
                    and not state.stopped_emitted
                    and self._can_emit(
                        state,
                        "OBJECT_STOPPED",
                        now,
                    )
                ):

                    confidence = self._stopped_confidence(
                        speed
                    )

                    events.append(
                        self._event(
                            event_type="OBJECT_STOPPED",
                            timestamp=now,
                            track_id=track_id,
                            object_id=object_id,
                            label=label,
                            confidence=confidence,
                        )
                    )

                    state.stopped_emitted = True

            else:

                # 物体再次运动
                state.stopped_since = None
                state.stopped_emitted = False

            # ==================================================
            # 4. PICK_UP
            #
            # 条件：
            #
            # 手靠近
            # +
            # 物体运动
            # +
            # 持续时间
            #
            # confidence：
            #
            # hand evidence
            # +
            # moving evidence
            # +
            # object detection confidence
            # ==================================================

            if nearby and speed >= self.moving_speed:

                if state.held_since is None:
                    state.held_since = now

                if (
                    self._persisted(
                        state.held_since,
                        now,
                    )
                    and not state.pickup_emitted
                    and self._can_emit(
                        state,
                        "PICK_UP",
                        now,
                    )
                ):

                    hand_confidence = (
                        self._distance_confidence(
                            hand_distance
                        )
                    )

                    moving_confidence = (
                        self._moving_confidence(
                            speed
                        )
                    )

                    pickup_confidence = (
                        hand_confidence
                        * self.pickup_hand_weight
                        +
                        moving_confidence
                        * self.pickup_moving_weight
                        +
                        object_confidence
                        * self.pickup_object_weight
                    )

                    pickup_confidence = self._bounded_confidence(
                        pickup_confidence,
                        self.pickup_conf_min,
                        self.pickup_conf_max,
                    )

                    events.append(
                        self._event(
                            event_type="PICK_UP",
                            timestamp=now,
                            track_id=track_id,
                            object_id=object_id,
                            label=label,
                            confidence=pickup_confidence,
                        )
                    )

                    # ------------------------------------------
                    # PICK_UP 成功
                    # ------------------------------------------

                    state.pickup_emitted = True
                    state.pickup_confidence = (
                        pickup_confidence
                    )

            elif not nearby:

                state.held_since = None

            # ==================================================
            # 5. PLACE
            #
            # 严格要求：
            #
            # PICK_UP
            # ↓
            # MOVING
            # ↓
            # STOPPED
            # ↓
            # HAND AWAY
            # ↓
            # PLACE
            #
            # 不允许：
            #
            # 物体自己移动
            # ↓
            # 停止
            # ↓
            # PLACE
            # ==================================================

            if (
                state.pickup_emitted
                and state.has_moved
                and state.stopped_since is not None
                and speed <= self.stopped_speed
                and not nearby
            ):

                stopped_duration_ms = (
                    now - state.stopped_since
                ) * 1000.0

                if (
                    stopped_duration_ms
                    >= self.persistence_ms
                    and self._can_emit(
                        state,
                        "PLACE",
                        now,
                    )
                ):

                    # ------------------------------------------
                    # Pickup evidence
                    # ------------------------------------------

                    pickup_confidence = (
                        state.pickup_confidence
                    )

                    # ------------------------------------------
                    # Moving evidence
                    # ------------------------------------------

                    moving_confidence = (
                        self._moving_confidence(
                            max(
                                speed,
                                self.moving_speed,
                            )
                        )
                    )

                    # ------------------------------------------
                    # Stopped evidence
                    # ------------------------------------------

                    stopped_confidence = (
                        self._stopped_confidence(
                            speed
                        )
                    )

                    # ------------------------------------------
                    # Hand-away evidence
                    #
                    # 手距离越远，越接近“已经放下”
                    # ------------------------------------------

                    hand_away_confidence = (
                        self._hand_away_confidence(
                            hand_distance
                        )
                    )

                    # ------------------------------------------
                    # 多证据融合
                    # ------------------------------------------

                    place_confidence = (
                        pickup_confidence
                        * self.place_pickup_weight
                        +
                        moving_confidence
                        * self.place_moving_weight
                        +
                        stopped_confidence
                        * self.place_stopped_weight
                        +
                        hand_away_confidence
                        * self.place_hand_away_weight
                    )

                    place_confidence = self._bounded_confidence(
                        place_confidence,
                        self.place_conf_min,
                        self.place_conf_max,
                    )

                    events.append(
                        self._event(
                            event_type="PLACE",
                            timestamp=now,
                            track_id=track_id,
                            object_id=object_id,
                            label=label,
                            confidence=place_confidence,
                        )
                    )

                    # ------------------------------------------
                    # PLACE 完成
                    #
                    # 开始新的动作周期
                    # ------------------------------------------

                    state.placed_since = now

                    state.stopped_since = None
                    state.moving_since = None

                    state.has_moved = False

                    state.pickup_emitted = False
                    state.pickup_confidence = 0.0

                    state.moving_emitted = False
                    state.stopped_emitted = False

                    state.held_since = None

        # ==================================================
        # 清理 stale tracks
        # ==================================================

        self._cleanup_stale_objects(
            now=now
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
            perception=perception
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
        根据当前 perception 判断当前通用状态。

        只判断：
            IDLE
            INTERACTING
            MOVING_OBJECT
        """

        objects = perception.get(
            "objects",
            [],
        ) or []

        relations = perception.get(
            "relations",
            [],
        ) or []

        # --------------------------------------------------
        # 没有物体
        # --------------------------------------------------

        if not objects:
            return "IDLE"

        # --------------------------------------------------
        # 手是否靠近物体
        # --------------------------------------------------

        hand_near = False

        for relation in relations:

            if relation.get("type") not in {
                "HAND_OBJECT_DISTANCE",
                "HAND_NEAR_OBJECT",
            }:
                continue

            if self._relation_is_near(
                relation
            ):
                hand_near = True
                break

        # --------------------------------------------------
        # 是否存在运动物体
        # --------------------------------------------------

        moving = False

        for obj in objects:

            motion = obj.get(
                "motion",
                {},
            ) or {}

            try:
                speed = float(
                    motion.get(
                        "speed",
                        0.0,
                    )
                )
            except (TypeError, ValueError):
                speed = 0.0

            if speed >= self.moving_speed:
                moving = True
                break

        # --------------------------------------------------
        # 状态
        # --------------------------------------------------

        if hand_near and moving:
            return "MOVING_OBJECT"

        if hand_near:
            return "INTERACTING"

        if moving:
            return "MOVING_OBJECT"

        return "IDLE"

    def _relation_is_near(
        self,
        relation: dict,
    ) -> bool:
        """根据活动配置判断手物距离是否达到 near 阈值。"""

        try:
            distance = float(
                relation.get(
                    "distance",
                    9999.0,
                )
            )
        except (TypeError, ValueError):
            return False

        if not math.isfinite(distance):
            return False

        return distance <= self.near_distance

    def _stabilize_state(
        self,
        raw_state: str,
        now: float,
    ) -> str:
        """
        Activity State 防抖。

        raw_state 不会因为单帧变化立即切换。
        """

        # --------------------------------------------------
        # 当前状态没有变化
        # --------------------------------------------------

        if raw_state == self.last_state:

            self.candidate_state = None
            self.candidate_since = None

            return self.last_state

        # --------------------------------------------------
        # 新候选状态
        # --------------------------------------------------

        if self.candidate_state != raw_state:

            self.candidate_state = raw_state
            self.candidate_since = now

            return self.last_state

        # --------------------------------------------------
        # 防止异常
        # --------------------------------------------------

        if self.candidate_since is None:

            self.candidate_since = now

            return self.last_state

        # --------------------------------------------------
        # 持续时间
        # --------------------------------------------------

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
    # Persistence
    # ======================================================

    def _persisted(
        self,
        since: float | None,
        now: float,
    ) -> bool:
        """判断状态是否持续足够长时间。"""

        return (
            since is not None
            and (
                now - since
            ) * 1000.0
            >= self.persistence_ms
        )

    # ======================================================
    # Event Deduplication
    # ======================================================

    def _event_already_active(
        self,
        state: ActiveObjectState,
        event_type: str,
    ) -> bool:
        """
        判断事件是否已经在当前动作周期中触发过。

        HAND_NEAR_OBJECT：
            一个靠近周期只触发一次。
        """

        if event_type == "HAND_NEAR_OBJECT":
            return (
                state.last_event_at.get(
                    event_type
                )
                is not None
                and state.near_since is not None
            )

        if event_type == "OBJECT_MOVING":
            return state.moving_emitted

        if event_type == "OBJECT_STOPPED":
            return state.stopped_emitted

        if event_type == "PICK_UP":
            return state.pickup_emitted

        return False

    def _can_emit(
        self,
        state: ActiveObjectState,
        event_type: str,
        now: float,
    ) -> bool:
        """
        Event cooldown。

        cooldown 只是第二层保护。

        真正防止重复触发的是：
            moving_emitted
            stopped_emitted
            pickup_emitted
            HAND_NEAR_OBJECT 的状态边沿
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

    @staticmethod
    def _clamp(
        value: float,
    ) -> float:
        """限制到 [0, 1]。"""

        return max(
            0.0,
            min(
                1.0,
                float(value),
            ),
        )

    @staticmethod
    def _bounded_confidence(
        value: float,
        minimum: float,
        maximum: float,
    ) -> float:
        """
        把 confidence 限制在指定区间。

        例如：
            pickup min = 0.65
            pickup max = 0.95

        则最终：
            0.50 -> 0.65
            0.80 -> 0.80
            0.99 -> 0.95
        """

        minimum = max(
            0.0,
            min(
                1.0,
                minimum,
            ),
        )

        maximum = max(
            minimum,
            min(
                1.0,
                maximum,
            ),
        )

        return max(
            minimum,
            min(
                maximum,
                float(value),
            ),
        )

    # ------------------------------------------------------
    # Hand Near
    # ------------------------------------------------------

    def _distance_confidence(
        self,
        distance: float | None,
    ) -> float:
        """
        手越靠近物体，confidence 越高。

        distance = 0
            -> hand_near_conf_max

        distance = near_distance
            -> hand_near_conf_min
        """

        if distance is None:
            return self.hand_near_conf_min

        try:
            distance = float(distance)
        except (TypeError, ValueError):
            return self.hand_near_conf_min

        if not math.isfinite(distance):
            return self.hand_near_conf_min

        distance = max(
            0.0,
            distance,
        )

        if distance >= self.near_distance:
            return self.hand_near_conf_min

        if self.near_distance <= 0:
            return self.hand_near_conf_max

        ratio = (
            distance
            / self.near_distance
        )

        confidence = (
            self.hand_near_conf_max
            - ratio
            * (
                self.hand_near_conf_max
                - self.hand_near_conf_min
            )
        )

        return self._clamp(
            confidence
        )

    # ------------------------------------------------------
    # Moving
    # ------------------------------------------------------

    def _moving_confidence(
        self,
        speed: float,
    ) -> float:
        """
        根据物体速度计算运动 confidence。

        moving_speed
            -> moving_conf_min

        strong_speed
            -> moving_conf_max
        """

        try:
            speed = float(speed)
        except (TypeError, ValueError):
            return self.moving_conf_min

        if not math.isfinite(speed):
            return self.moving_conf_min

        if speed <= self.moving_speed:
            return self.moving_conf_min

        if (
            self.moving_strong_speed
            <= self.moving_speed
        ):
            return self.moving_conf_max

        ratio = (
            speed - self.moving_speed
        ) / (
            self.moving_strong_speed
            - self.moving_speed
        )

        ratio = self._clamp(
            ratio
        )

        confidence = (
            self.moving_conf_min
            + ratio
            * (
                self.moving_conf_max
                - self.moving_conf_min
            )
        )

        return self._clamp(
            confidence
        )

    # ------------------------------------------------------
    # Stopped
    # ------------------------------------------------------

    def _stopped_confidence(
        self,
        speed: float,
    ) -> float:
        """
        速度越接近 0，停止 confidence 越高。

        speed = 0
            -> stopped_conf_max

        speed = stopped_speed
            -> stopped_conf_min
        """

        try:
            speed = float(speed)
        except (TypeError, ValueError):
            return self.stopped_conf_min

        if not math.isfinite(speed):
            return self.stopped_conf_min

        speed = max(
            0.0,
            speed,
        )

        if speed >= self.stopped_speed:
            return self.stopped_conf_min

        if self.stopped_speed <= 0:
            return self.stopped_conf_max

        ratio = (
            speed
            / self.stopped_speed
        )

        confidence = (
            self.stopped_conf_max
            - ratio
            * (
                self.stopped_conf_max
                - self.stopped_conf_min
            )
        )

        return self._clamp(
            confidence
        )

    # ------------------------------------------------------
    # Hand Away
    # ------------------------------------------------------

    def _hand_away_confidence(
        self,
        distance: float | None,
    ) -> float:
        """
        计算“手已经离开物体”的 confidence。

        逻辑：

            distance <= near_distance
                -> 低

            distance >= 2 * near_distance
                -> 高
        """

        if distance is None:
            return 0.95

        try:
            distance = float(distance)
        except (TypeError, ValueError):
            return 0.80

        if not math.isfinite(distance):
            return 0.80

        if distance <= self.near_distance:
            return 0.20

        strong_away_distance = (
            self.near_distance * 2.0
        )

        if distance >= strong_away_distance:
            return 0.95

        ratio = (
            distance
            - self.near_distance
        ) / (
            strong_away_distance
            - self.near_distance
        )

        return (
            0.20
            + ratio * 0.75
        )

    # ======================================================
    # Object Cleanup
    # ======================================================

    def _cleanup_stale_objects(
        self,
        now: float,
    ) -> None:
        """
        清理长时间没有出现在 perception 中的 Track。

        当前只清理内部状态，
        不产生 OBJECT_DISAPPEARED Event。
        """

        stale_ids = [
            track_id
            for track_id, state
            in self.objects.items()
            if (
                now - state.last_seen
                > self.stale_timeout_seconds
            )
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

            try:
                return (
                    float(bbox[0]),
                    float(bbox[1]),
                    float(bbox[2]),
                    float(bbox[3]),
                )
            except (TypeError, ValueError):
                return None

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
                    self._clamp(
                        confidence
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