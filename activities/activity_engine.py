from collections import deque
import math
from dataclasses import dataclass, field
from pathlib import Path
import time
from typing import Any

import yaml


@dataclass
class ActiveObjectState:
    """
    单个被追踪物体在 Activity Engine 中的运行状态。

    注意：
    ActivityEngine 只负责通用行为，不负责具体活动语义。

    例如：
        STACKED
        STRUCTURE_OK
        STEP_COMPLETED
        ACTIVITY_COMPLETED

    不在这里判断。
    """

    track_id: int
    object_id: str
    label: str

    last_center: tuple[float, float]

    last_speed: float = 0.0
    last_seen: float = 0.0

    # ==========================================================
    # 当前通用状态
    # ==========================================================

    # 手是否处于物体附近
    near_active: bool = False

    # 物体是否处于运动状态
    moving_active: bool = False

    # 物体是否处于停止状态
    stopped_active: bool = False

    # ==========================================================
    # 状态持续时间
    # ==========================================================

    near_since: float | None = None
    moving_since: float | None = None
    stopped_since: float | None = None

    # ==========================================================
    # 当前交互周期
    # ==========================================================

    # 是否经历过有效运动
    has_moved: bool = False

    # 当前交互周期是否已经触发 PICK_UP
    pickup_emitted: bool = False

    # 当前运动周期是否已经触发 OBJECT_MOVING
    moving_emitted: bool = False

    # 当前停止周期是否已经触发 OBJECT_STOPPED
    stopped_emitted: bool = False

    # 当前放置周期是否已经触发 PLACE
    place_emitted: bool = False

    # ==========================================================
    # PICK_UP 持续判断
    # ==========================================================

    held_since: float | None = None

    # ==========================================================
    # 最近一次 PLACE
    # ==========================================================

    placed_since: float | None = None

    # ==========================================================
    # Event 最近触发时间
    #
    # 这是第二层保险。
    #
    # 真正的“只触发一次”由上面的 *_emitted 状态负责。
    # ==========================================================

    last_event_at: dict[str, float] = field(default_factory=dict)


class ActivityEngine:
    """
    通用 Activity Engine。

    ------------------------------------------------------------
    职责
    ------------------------------------------------------------

    1. 接收 Vision Layer 输出的标准化 perception JSON
    2. 根据时间窗口进行基础行为判断
    3. 维护稳定的通用行为状态
    4. 输出通用 Activity Events

    ------------------------------------------------------------
    输出的通用 Event
    ------------------------------------------------------------

        OBJECT_APPEARED
        HAND_NEAR_OBJECT
        OBJECT_MOVING
        OBJECT_STOPPED
        PICK_UP
        PLACE

    ------------------------------------------------------------
    不负责
    ------------------------------------------------------------

        STACKED
        STRUCTURE_OK
        STEP_COMPLETED
        ACTIVITY_COMPLETED

    这些高级活动语义交给：

        Activity Understanding / VLM

    ------------------------------------------------------------
    核心设计
    ------------------------------------------------------------

    状态 State 和事件 Event 分离。

    例如：

        HAND_NEAR_OBJECT

    实际上表示：

        “当前手仍然在物体附近”

    但 Event 只在：

        非附近 -> 附近

    这个状态边沿发生时触发一次。

    同理：

        OBJECT_MOVING

    只在：

        非运动 -> 运动

    的时候触发一次。

    这样可以避免：

        OBJECT_MOVING
        OBJECT_MOVING
        OBJECT_MOVING
        OBJECT_MOVING

    以及：

        PICK_UP
        PICK_UP
        PICK_UP

    这种重复事件。
    """

    def __init__(
        self,
        config_path: str | Path,
        window_seconds: float = 2.0,
    ):
        self.config_path = Path(config_path)

        raw = yaml.safe_load(
            self.config_path.read_text(
                encoding="utf-8"
            )
        ) or {}

        self.config = raw.get(
            "activity",
            raw,
        )

        self.activity_id = self.config.get(
            "id",
            self.config_path.stem,
        )

        self.activity_name = self.config.get(
            "name",
            self.activity_id,
        )

        self.window_seconds = float(
            window_seconds
        )

        # ======================================================
        # 时间历史
        # ======================================================

        # 假设最大 60 FPS，预留双倍空间
        max_history_len = max(
            1,
            int(self.window_seconds * 120),
        )

        self.history: deque[dict[str, Any]] = deque(
            maxlen=max_history_len
        )

        # ======================================================
        # 当前物体状态
        # ======================================================

        self.objects: dict[int, ActiveObjectState] = {}

        # ======================================================
        # Event 统计
        # ======================================================

        self.event_counter = 0
        self.total_event_count = 0

        # ======================================================
        # Activity State
        # ======================================================

        self.last_state = "IDLE"

        self.candidate_state: str | None = None
        self.candidate_since: float | None = None

        # ======================================================
        # Activity State 防抖时间
        # ======================================================

        default_state_persistence = {
            "IDLE": 500.0,
            "INTERACTING": 300.0,
            "MOVING_OBJECT": 300.0,
        }

        custom_state_persistence = (
            self.config.get(
                "state_persistence_ms",
                {},
            )
            or {}
        )

        self.state_persistence_ms = {
            **default_state_persistence,
            **custom_state_persistence,
        }

        # ======================================================
        # 通用 Engine 参数
        # ======================================================

        defaults = (
            self.config.get(
                "engine",
                {},
            )
            or {}
        )

        # ------------------------------------------------------
        # 手-物体接近距离
        # ------------------------------------------------------

        self.near_distance = float(
            defaults.get(
                "hand_near_distance_px",
                120.0,
            )
        )

        # ------------------------------------------------------
        # 运动阈值
        # ------------------------------------------------------

        self.moving_speed = float(
            defaults.get(
                "moving_speed_px_s",
                45.0,
            )
        )

        # ------------------------------------------------------
        # 停止阈值
        #
        # 当前项目已经验证 30 比 18 更稳定。
        # ------------------------------------------------------

        self.stopped_speed = float(
            defaults.get(
                "stopped_speed_px_s",
                30.0,
            )
        )

        # ------------------------------------------------------
        # 状态 / 事件持续时间
        # ------------------------------------------------------

        self.persistence_ms = float(
            defaults.get(
                "persistence_ms",
                250.0,
            )
        )

        # ------------------------------------------------------
        # Event cooldown
        #
        # 现在不是核心去重机制。
        # 只是作为第二层保险。
        # ------------------------------------------------------

        self.event_cooldown_ms = float(
            defaults.get(
                "event_cooldown_ms",
                450.0,
            )
        )

        # ------------------------------------------------------
        # Stale Track 清理
        # ------------------------------------------------------

        self.stale_timeout_seconds = float(
            defaults.get(
                "stale_timeout_seconds",
                self.window_seconds * 2.0,
            )
        )

        # ------------------------------------------------------
        # Confidence
        # ------------------------------------------------------

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

    # ==========================================================
    # Public API
    # ==========================================================

    def reset(self) -> None:
        """
        重置 Activity Engine。
        """

        self.history.clear()
        self.objects.clear()

        self.event_counter = 0
        self.total_event_count = 0

        self.last_state = "IDLE"

        self.candidate_state = None
        self.candidate_since = None

    def update(
        self,
        perception: dict,
    ) -> list[dict]:
        """
        输入一帧 perception。

        返回：
            这一帧新产生的 Events。

        注意：
            State 会持续维护在内部；
            Event 只在状态边沿发生时输出。
        """

        now = float(
            perception.get(
                "timestamp",
                time.time(),
            )
        )

        # ======================================================
        # 保存时间窗口
        # ======================================================

        self.history.append(perception)

        while (
            self.history
            and
            now
            - float(
                self.history[0].get(
                    "timestamp",
                    now,
                )
            )
            > self.window_seconds
        ):
            self.history.popleft()

        # ======================================================
        # 获取 Vision 数据
        # ======================================================

        objects = (
            perception.get(
                "objects",
                [],
            )
            or []
        )

        relations = (
            perception.get(
                "relations",
                [],
            )
            or []
        )

        # ======================================================
        # 建立手-物体距离映射
        #
        # Serializer：
        #
        #     HAND_OBJECT_DISTANCE
        #
        # ActivityEngine：
        #
        #     根据 YAML 判断 HAND_NEAR_OBJECT
        #
        # 同时兼容旧：
        #
        #     HAND_NEAR_OBJECT
        # ======================================================

        relation_map: dict[int, float] = {}

        for relation in relations:

            relation_type = relation.get(
                "type"
            )

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
            except (
                TypeError,
                ValueError,
            ):
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
            except (
                TypeError,
                ValueError,
            ):
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

        # ======================================================
        # 处理所有物体
        # ======================================================

        events: list[dict] = []

        for obj in objects:

            try:
                track_id = int(
                    obj.get(
                        "track_id",
                        -1,
                    )
                )
            except (
                TypeError,
                ValueError,
            ):
                continue

            if track_id < 0:
                continue

            # ==================================================
            # 基础物体信息
            # ==================================================

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

            center = (
                obj.get(
                    "center",
                    {},
                )
                or {}
            )

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
            except (
                TypeError,
                ValueError,
            ):
                center_xy = (0.0, 0.0)

            motion = (
                obj.get(
                    "motion",
                    {},
                )
                or {}
            )

            try:
                speed = float(
                    motion.get(
                        "speed",
                        0.0,
                    )
                )
            except (
                TypeError,
                ValueError,
            ):
                speed = 0.0

            if not math.isfinite(speed):
                speed = 0.0

            # ==================================================
            # 获取 / 创建物体状态
            # ==================================================

            state = self.objects.get(
                track_id
            )

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

                # ----------------------------------------------
                # OBJECT_APPEARED
                #
                # 新 Track 只触发一次
                # ----------------------------------------------

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

            # ==================================================
            # 更新基础信息
            # ==================================================

            state.object_id = object_id
            state.label = label
            state.last_center = center_xy
            state.last_speed = speed
            state.last_seen = now

            # ==================================================
            # 当前手是否靠近
            # ==================================================

            hand_distance = relation_map.get(
                track_id
            )

            nearby = (
                hand_distance is not None
                and
                hand_distance <= self.near_distance
            )

            # ==================================================
            # 1. HAND_NEAR_OBJECT
            #
            # 这是“状态边沿事件”。
            #
            # False -> True：
            #
            #     触发一次
            #
            # True -> True：
            #
            #     不再触发
            #
            # True -> False：
            #
            #     状态结束
            # ==================================================

            self._update_near_state(
                state=state,
                nearby=nearby,
                now=now,
                hand_distance=hand_distance,
                events=events,
                object_id=object_id,
                label=label,
                track_id=track_id,
            )

            # ==================================================
            # 2. OBJECT_MOVING
            #
            # 只有：
            #
            #     非运动 -> 运动
            #
            # 才产生一次 Event。
            # ==================================================

            if speed >= self.moving_speed:

                self._handle_moving(
                    state=state,
                    now=now,
                    speed=speed,
                    events=events,
                    object_id=object_id,
                    label=label,
                    track_id=track_id,
                )

            # ==================================================
            # 3. OBJECT_STOPPED
            #
            # 只有：
            #
            #     MOVING -> STOPPED
            #
            # 才产生一次 Event。
            # ==================================================

            elif speed <= self.stopped_speed:

                self._handle_stopped(
                    state=state,
                    now=now,
                    speed=speed,
                    events=events,
                    object_id=object_id,
                    label=label,
                    track_id=track_id,
                )

            # ==================================================
            # 4. PICK_UP
            #
            # 条件：
            #
            #     手靠近
            #     +
            #     物体正在运动
            #     +
            #     当前交互周期尚未 PICK_UP
            #
            # 一个交互周期只触发一次。
            # ==================================================

            if nearby and speed >= self.moving_speed:

                self._handle_pickup(
                    state=state,
                    now=now,
                    speed=speed,
                    events=events,
                    object_id=object_id,
                    label=label,
                    track_id=track_id,
                )

            # ==================================================
            # 5. PLACE
            #
            # 条件：
            #
            #     物体之前移动过
            #     +
            #     当前已经停止
            #     +
            #     手已经离开
            #
            # 一个停止/放置周期只触发一次。
            # ==================================================

            if (
                speed <= self.stopped_speed
                and
                state.stopped_since is not None
                and
                state.has_moved
                and
                not nearby
            ):

                self._handle_place(
                    state=state,
                    now=now,
                    events=events,
                    object_id=object_id,
                    label=label,
                    track_id=track_id,
                )

        # ======================================================
        # 清理 stale Track
        # ======================================================

        self._cleanup_stale_objects(
            now=now
        )

        # ======================================================
        # Event 统计
        # ======================================================

        if events:
            self.total_event_count += len(
                events
            )

        # ======================================================
        # Activity State
        # ======================================================

        raw_state = self._derive_raw_state(
            perception=perception
        )

        self.last_state = self._stabilize_state(
            raw_state=raw_state,
            now=now,
        )

        return events

    # ==========================================================
    # HAND NEAR
    # ==========================================================

    def _update_near_state(
        self,
        *,
        state: ActiveObjectState,
        nearby: bool,
        now: float,
        hand_distance: float | None,
        events: list[dict],
        object_id: str,
        label: str,
        track_id: int,
    ) -> None:
        """
        更新 HAND_NEAR_OBJECT 状态。

        核心原则：

            False -> True
                触发一次

            True -> True
                不触发

            True -> False
                结束当前 near 状态

        """

        # ------------------------------------------------------
        # 手进入附近
        # ------------------------------------------------------

        if nearby:

            if not state.near_active:

                state.near_active = True
                state.near_since = now

                # ----------------------------------------------
                # 只在第一次进入时触发
                # ----------------------------------------------

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

            elif state.near_since is None:

                state.near_since = now

            return

        # ------------------------------------------------------
        # 手离开
        # ------------------------------------------------------

        if state.near_active:

            state.near_active = False
            state.near_since = None

            # --------------------------------------------------
            # 当前交互周期结束。
            #
            # 下一次手重新靠近时：
            #
            #     可以重新 PICK_UP
            #
            # --------------------------------------------------

            state.held_since = None

            # --------------------------------------------------
            # 如果物体已经停止，
            # 当前完整动作周期可以视为结束。
            # --------------------------------------------------

            if state.stopped_active:

                state.pickup_emitted = False
                state.moving_emitted = False
                state.stopped_emitted = False
                state.place_emitted = False

            return

    # ==========================================================
    # MOVING
    # ==========================================================

    def _handle_moving(
        self,
        *,
        state: ActiveObjectState,
        now: float,
        speed: float,
        events: list[dict],
        object_id: str,
        label: str,
        track_id: int,
    ) -> None:
        """
        处理 OBJECT_MOVING。

        只有：

            非运动 -> 运动

        才触发一次。
        """

        # ------------------------------------------------------
        # 第一次进入运动状态
        # ------------------------------------------------------

        if not state.moving_active:

            state.moving_active = True

            state.moving_since = now

            # 进入运动后，停止状态失效
            state.stopped_active = False
            state.stopped_since = None

            # 记录真实运动
            state.has_moved = True

        else:

            # 已经运动，只保持状态
            if state.moving_since is None:
                state.moving_since = now

            state.has_moved = True

            state.stopped_active = False
            state.stopped_since = None

        # ------------------------------------------------------
        # persistence
        # ------------------------------------------------------

        if not self._persisted(
            state.moving_since,
            now,
        ):
            return

        # ------------------------------------------------------
        # 已经发送过 OBJECT_MOVING
        #
        # 不再发送。
        # ------------------------------------------------------

        if state.moving_emitted:
            return

        # ------------------------------------------------------
        # 第二层 cooldown
        # ------------------------------------------------------

        if not self._can_emit(
            state,
            "OBJECT_MOVING",
            now,
        ):
            return

        # ------------------------------------------------------
        # 触发 Event
        # ------------------------------------------------------

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

        state.moving_emitted = True

    # ==========================================================
    # STOPPED
    # ==========================================================

    def _handle_stopped(
        self,
        *,
        state: ActiveObjectState,
        now: float,
        speed: float,
        events: list[dict],
        object_id: str,
        label: str,
        track_id: int,
    ) -> None:
        """
        处理 OBJECT_STOPPED。

        必须：

            之前经历过有效运动
            +
            速度持续低于 stopped_speed

        才认为是有效停止。

        只触发一次。
        """

        # ------------------------------------------------------
        # 第一次进入低速区间
        # ------------------------------------------------------

        if state.stopped_since is None:

            state.stopped_since = now

        # ------------------------------------------------------
        # 进入停止状态
        #
        # 只有之前正在运动，
        # 才能形成：
        #
        #     MOVING -> STOPPED
        # ------------------------------------------------------

        if state.moving_active:

            stopped_duration_ms = (
                now - state.stopped_since
            ) * 1000.0

            if stopped_duration_ms < self.persistence_ms:
                return

            # --------------------------------------------------
            # 状态切换
            # --------------------------------------------------

            state.moving_active = False
            state.stopped_active = True

        else:

            # --------------------------------------------------
            # 如果从来没有运动过，
            # 不产生 OBJECT_STOPPED。
            # --------------------------------------------------

            if not state.has_moved:
                return

            state.stopped_active = True

        # ------------------------------------------------------
        # 已经发送过 STOPPED
        # ------------------------------------------------------

        if state.stopped_emitted:
            return

        # ------------------------------------------------------
        # 第二层 cooldown
        # ------------------------------------------------------

        if not self._can_emit(
            state,
            "OBJECT_STOPPED",
            now,
        ):
            return

        # ------------------------------------------------------
        # 触发 Event
        # ------------------------------------------------------

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

        state.stopped_emitted = True

    # ==========================================================
    # PICK UP
    # ==========================================================

    def _handle_pickup(
        self,
        *,
        state: ActiveObjectState,
        now: float,
        speed: float,
        events: list[dict],
        object_id: str,
        label: str,
        track_id: int,
    ) -> None:
        """
        处理 PICK_UP。

        条件：

            手靠近
            +
            物体运动
            +
            当前交互周期还没有 PICK_UP

        一个交互周期只触发一次。
        """

        # ------------------------------------------------------
        # 当前周期已经 PICK_UP
        # ------------------------------------------------------

        if state.pickup_emitted:
            return

        # ------------------------------------------------------
        # 第一次进入“疑似拿起”
        # ------------------------------------------------------

        if state.held_since is None:

            state.held_since = now

        # ------------------------------------------------------
        # 必须持续一定时间
        # ------------------------------------------------------

        if not self._persisted(
            state.held_since,
            now,
        ):
            return

        # ------------------------------------------------------
        # 第二层 cooldown
        # ------------------------------------------------------

        if not self._can_emit(
            state,
            "PICK_UP",
            now,
        ):
            return

        # ------------------------------------------------------
        # 触发 PICK_UP
        # ------------------------------------------------------

        events.append(
            self._event(
                event_type="PICK_UP",
                timestamp=now,
                track_id=track_id,
                object_id=object_id,
                label=label,
                confidence=min(
                    1.0,
                    0.55
                    + speed
                    / max(
                        self.pickup_speed_scale,
                        1.0,
                    ),
                ),
            )
        )

        # ------------------------------------------------------
        # 锁住当前周期
        # ------------------------------------------------------

        state.pickup_emitted = True

    # ==========================================================
    # PLACE
    # ==========================================================

    def _handle_place(
        self,
        *,
        state: ActiveObjectState,
        now: float,
        events: list[dict],
        object_id: str,
        label: str,
        track_id: int,
    ) -> None:
        """
        处理 PLACE。

        条件：

            物体之前移动过
            +
            当前停止
            +
            手已经离开

        一个完整放置周期只触发一次。
        """

        # ------------------------------------------------------
        # 当前周期已经 PLACE
        # ------------------------------------------------------

        if state.place_emitted:
            return

        # ------------------------------------------------------
        # 第二层 cooldown
        # ------------------------------------------------------

        if not self._can_emit(
            state,
            "PLACE",
            now,
        ):
            return

        # ------------------------------------------------------
        # 触发 PLACE
        # ------------------------------------------------------

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

        state.place_emitted = True
        state.placed_since = now

        # ------------------------------------------------------
        # 完整动作周期结束
        #
        # 注意：
        #
        # 不马上清除 pickup_emitted，
        # 因为手可能还没有重新接近。
        #
        # 只有下一次新的 near 周期才重新允许 PICK_UP。
        # ------------------------------------------------------

        state.stopped_since = None
        state.moving_since = None

        state.has_moved = False

        state.moving_active = False
        state.stopped_active = False

        state.stopped_emitted = False
        state.moving_emitted = False

    # ==========================================================
    # Activity State
    # ==========================================================

    def _derive_raw_state(
        self,
        perception: dict,
    ) -> str:
        """
        根据当前 perception 判断通用 Activity State。

        只输出：

            IDLE
            INTERACTING
            MOVING_OBJECT

        不判断具体活动语义。
        """

        objects = (
            perception.get(
                "objects",
                [],
            )
            or []
        )

        relations = (
            perception.get(
                "relations",
                [],
            )
            or []
        )

        # ------------------------------------------------------
        # 没有物体
        # ------------------------------------------------------

        if not objects:
            return "IDLE"

        # ------------------------------------------------------
        # 手是否靠近
        # ------------------------------------------------------

        hand_near = any(
            relation.get("type")
            in {
                "HAND_OBJECT_DISTANCE",
                "HAND_NEAR_OBJECT",
            }
            and self._relation_is_near(
                relation
            )
            for relation in relations
        )

        # ------------------------------------------------------
        # 是否存在运动物体
        # ------------------------------------------------------

        moving = False

        for obj in objects:

            motion = (
                obj.get(
                    "motion",
                    {},
                )
                or {}
            )

            try:
                speed = float(
                    motion.get(
                        "speed",
                        0.0,
                    )
                )
            except (
                TypeError,
                ValueError,
            ):
                speed = 0.0

            if (
                math.isfinite(speed)
                and
                speed >= self.moving_speed
            ):
                moving = True
                break

        # ------------------------------------------------------
        # 状态判断
        # ------------------------------------------------------

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
        """
        根据当前活动配置判断
        手-物体距离是否属于“靠近”。
        """

        try:
            distance = float(
                relation.get(
                    "distance",
                    9999.0,
                )
            )
        except (
            TypeError,
            ValueError,
        ):
            return False

        if not math.isfinite(distance):
            return False

        return (
            distance
            <= self.near_distance
        )

    def _stabilize_state(
        self,
        raw_state: str,
        now: float,
    ) -> str:
        """
        Activity State 防抖。

        不因为单帧变化立即切换。
        """

        # ------------------------------------------------------
        # 当前状态没有变化
        # ------------------------------------------------------

        if raw_state == self.last_state:

            self.candidate_state = None
            self.candidate_since = None

            return self.last_state

        # ------------------------------------------------------
        # 新候选状态
        # ------------------------------------------------------

        if self.candidate_state != raw_state:

            self.candidate_state = raw_state
            self.candidate_since = now

            return self.last_state

        # ------------------------------------------------------
        # 没有候选开始时间
        # ------------------------------------------------------

        if self.candidate_since is None:

            self.candidate_since = now

            return self.last_state

        # ------------------------------------------------------
        # 候选状态持续时间
        # ------------------------------------------------------

        duration_ms = (
            now - self.candidate_since
        ) * 1000.0

        required_ms = float(
            self.state_persistence_ms.get(
                raw_state,
                400.0,
            )
        )

        # ------------------------------------------------------
        # 状态切换
        # ------------------------------------------------------

        if duration_ms >= required_ms:

            self.last_state = raw_state

            self.candidate_state = None
            self.candidate_since = None

        return self.last_state

    # ==========================================================
    # Event 工具
    # ==========================================================

    def _persisted(
        self,
        since: float | None,
        now: float,
    ) -> bool:
        """
        判断某个状态是否持续足够长时间。
        """

        return (
            since is not None
            and
            (
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
        第二层 Event cooldown。

        注意：

        真正的事件去重由：

            pickup_emitted
            moving_emitted
            stopped_emitted
            place_emitted

        等状态锁完成。

        这里仅作为额外保护。
        """

        last = state.last_event_at.get(
            event_type
        )

        if last is not None:

            elapsed_ms = (
                now - last
            ) * 1000.0

            if (
                elapsed_ms
                < self.event_cooldown_ms
            ):
                return False

        state.last_event_at[event_type] = now

        return True

    # ==========================================================
    # Confidence
    # ==========================================================

    def _distance_confidence(
        self,
        distance: float | None,
    ) -> float:
        """
        根据手-物体距离计算接近置信度。
        """

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
        """
        根据运动速度计算运动置信度。
        """

        return max(
            0.0,
            min(
                1.0,
                0.55
                + speed
                / max(
                    self.moving_speed_scale,
                    1.0,
                ),
            ),
        )

    def _stopped_confidence(
        self,
        speed: float,
    ) -> float:
        """
        根据当前速度计算停止置信度。
        """

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

    # ==========================================================
    # Object Cleanup
    # ==========================================================

    def _cleanup_stale_objects(
        self,
        now: float,
    ) -> None:
        """
        清理长时间没有出现的 Track。

        当前只清理内部状态。

        不产生：

            OBJECT_DISAPPEARED
        """

        stale_ids = []

        for track_id, state in self.objects.items():

            if (
                now - state.last_seen
                > self.stale_timeout_seconds
            ):
                stale_ids.append(
                    track_id
                )

        for track_id in stale_ids:

            self.objects.pop(
                track_id,
                None,
            )

    # ==========================================================
    # BBox 工具
    # ==========================================================

    @staticmethod
    def _get_bbox(
        obj: dict,
    ) -> tuple[
        float,
        float,
        float,
        float,
    ] | None:
        """
        获取物体 BBox。

        当前 ActivityEngine 核心逻辑暂时不依赖 BBox，
        但保留该工具，方便后续 Activity Understanding
        或通用规则扩展。
        """

        bbox = obj.get("bbox")

        if (
            isinstance(
                bbox,
                (list, tuple),
            )
            and
            len(bbox) >= 4
        ):
            try:
                return (
                    float(bbox[0]),
                    float(bbox[1]),
                    float(bbox[2]),
                    float(bbox[3]),
                )
            except (
                TypeError,
                ValueError,
            ):
                return None

        return None

    # ==========================================================
    # Event 构造
    # ==========================================================

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
        """
        构造统一 Activity Event。
        """

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

            event["event"]["object"] = (
                object_data
            )

        return event