from __future__ import annotations

import json
import queue
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any

from config.model_config import (
    VLM_COOLDOWN,
    VLM_MODEL,
    VLM_TRIGGER_EVENTS,
)
from vision.vlm_client import QwenVLMClient


class ActivityUnderstanding:
    """将 ActivityEngine 通用事件交给 Qwen-VL 做高级活动语义理解。"""

    def __init__(
        self,
        activity_id: str,
        activity_name: str,
        model: str | None = None,
        cooldown_seconds: float | None = None,
        trigger_events: set[str] | None = None,
    ) -> None:
        self.activity_id = activity_id
        self.activity_name = activity_name
        self.cooldown_seconds = max(
            0.0,
            float(VLM_COOLDOWN if cooldown_seconds is None else cooldown_seconds),
        )
        self.trigger_events = set(trigger_events) if trigger_events is not None else set(VLM_TRIGGER_EVENTS)

        self.client = QwenVLMClient(model=model or VLM_MODEL)
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="qwen-vlm")
        self.results: queue.Queue[dict[str, Any]] = queue.Queue()
        self.lock = threading.Lock()
        self.last_submit_at = 0.0
        self.in_flight = False
        self.request_count = 0

    def _build_prompt(self, event: dict, perception: dict) -> str:
        event_data = event.get("event", {}) if isinstance(event, dict) else {}
        compact_perception = {
            "activity": perception.get("activity", {}),
            "frame": perception.get("frame", {}),
            "hands": [
                {
                    "hand_id": h.get("hand_id"),
                    "confidence": h.get("confidence"),
                    "wrist": h.get("wrist"),
                }
                for h in perception.get("hands", [])[:2]
            ],
            "objects": perception.get("objects", [])[:12],
            "relations": perception.get("relations", [])[:20],
        }

        return (
            "当前活动是：" + self.activity_name + "（" + self.activity_id + "）。\n"
            "ActivityEngine 刚刚产生了一个通用事件：\n"
            + json.dumps(event_data, ensure_ascii=False, indent=2)
            + "\n\n当前机器视觉感知 JSON：\n"
            + json.dumps(compact_perception, ensure_ascii=False, indent=2)
            + "\n\n请结合当前图片、通用事件和感知 JSON 做高级活动理解。\n"
            "不要把 ActivityEngine 的事件名称直接当成活动语义结论。"
            "例如 PLACE 只表示发生了放置行为，你需要观察图片判断放在哪里、是否与其他物体形成关系。\n"
            "只输出一个 JSON 对象，不要 Markdown，不要解释 JSON 之外的内容。JSON 字段必须包含：\n"
            "scene_summary: 当前画面的简短描述；\n"
            "recognized_objects: 图片中与当前活动有关的物体列表；\n"
            "relations: 物体之间的重要空间关系，例如 ON_TOP_OF、INSIDE、NEXT_TO；\n"
            "activity_step: 当前活动步骤；\n"
            "activity_completed: true/false/unknown；\n"
            "evidence: 支持判断的视觉证据；\n"
            "confidence: 0 到 1 的整体判断置信度。\n"
            "如果图片无法确认某个结论，请使用 unknown，而不是猜测。"
        )

    def _run(self, frame, event: dict, perception: dict, submitted_at: float) -> None:
        try:
            result = self.client.analyze(
                frame,
                self._build_prompt(event, perception),
            )
            self.results.put({
                "ok": True,
                "timestamp": time.time(),
                "submitted_at": submitted_at,
                "event": event,
                "result": result,
            })
        except Exception as exc:
            self.results.put({
                "ok": False,
                "timestamp": time.time(),
                "submitted_at": submitted_at,
                "event": event,
                "error": str(exc),
            })
        finally:
            with self.lock:
                self.in_flight = False

    def submit(self, frame, event: dict, perception: dict) -> bool:
        event_type = str(event.get("event", {}).get("type", ""))
        if event_type not in self.trigger_events:
            return False

        now = time.perf_counter()
        with self.lock:
            if self.in_flight:
                return False
            if now - self.last_submit_at < self.cooldown_seconds:
                return False
            self.last_submit_at = now
            self.in_flight = True
            self.request_count += 1
            request_id = self.request_count

        frame_copy = frame.copy()
        future: Future = self.executor.submit(
            self._run,
            frame_copy,
            event,
            perception,
            time.time(),
        )
        future.request_id = request_id  # type: ignore[attr-defined]
        return True

    def poll_results(self) -> list[dict[str, Any]]:
        output: list[dict[str, Any]] = []
        while True:
            try:
                output.append(self.results.get_nowait())
            except queue.Empty:
                break
        return output

    def close(self) -> None:
        self.executor.shutdown(wait=False, cancel_futures=True)
