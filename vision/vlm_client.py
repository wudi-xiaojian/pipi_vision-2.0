from __future__ import annotations

import base64
import json
import os
import re
from typing import Any

from config.model_config import (
    VLM_API_KEY_ENV,
    VLM_BASE_URL,
    VLM_MODEL,
    VLM_TIMEOUT,
)


class QwenVLMClient:
    """
    Qwen-VL 单帧视觉理解客户端。

    使用阿里云百炼 OpenAI-compatible API。

    当前北京 Workspace:
        ws-kotxcen8ue9z794f

    Endpoint:
        https://ws-kotxcen8ue9z794f.cn-beijing.maas.aliyuncs.com/compatible-mode/v1

    注意：
        API Key 不写死在代码里。
        从 DASHSCOPE_API_KEY 环境变量读取。
    """

    # ============================================================
    # 阿里云百炼：华北2（北京）
    # ============================================================

    BASE_URL = VLM_BASE_URL
    DEFAULT_MODEL = VLM_MODEL

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        timeout: float | None = None,
        base_url: str | None = None,
    ) -> None:
        self.model = model or VLM_MODEL
        self.api_key = api_key or os.getenv(VLM_API_KEY_ENV)
        self.timeout = float(VLM_TIMEOUT if timeout is None else timeout)
        self.base_url = base_url or VLM_BASE_URL

        # --------------------------------------------------------
        # API Key
        # --------------------------------------------------------

        if not self.api_key:
            raise RuntimeError(
                f"未找到 {VLM_API_KEY_ENV}。\n"
                "请先设置环境变量，例如：\n"
                f"export {VLM_API_KEY_ENV}=\"你的API Key\""
            )

        # --------------------------------------------------------
        # OpenAI SDK
        # --------------------------------------------------------

        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "未安装 openai SDK。\n"
                "请执行：\n"
                "pip install -U openai"
            ) from exc

        # --------------------------------------------------------
        # 创建 OpenAI-compatible Client
        # --------------------------------------------------------

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout,
        )

    # ============================================================
    # OpenCV BGR -> JPEG Data URL
    # ============================================================

    @staticmethod
    def _image_to_data_url(frame) -> str:
        """
        OpenCV BGR frame -> JPEG data URL。

        这样不需要把当前图片上传到 OSS，
        直接将当前摄像头帧作为 base64 图片发送给 Qwen-VL。
        """

        import cv2

        ok, encoded = cv2.imencode(
            ".jpg",
            frame,
            [
                int(cv2.IMWRITE_JPEG_QUALITY),
                85,
            ],
        )

        if not ok:
            raise RuntimeError(
                "无法将当前摄像头帧编码为 JPEG"
            )

        b64 = base64.b64encode(
            encoded.tobytes()
        ).decode("ascii")

        return f"data:image/jpeg;base64,{b64}"

    # ============================================================
    # 提取模型文本
    # ============================================================

    @staticmethod
    def _extract_text(response: Any) -> str:
        """
        从 OpenAI-compatible ChatCompletion 中提取文本。
        """

        try:
            content = response.choices[0].message.content
        except Exception as exc:
            raise RuntimeError(
                f"无法解析 Qwen-VL 响应: {response!r}"
            ) from exc

        if content is None:
            return ""

        # OpenAI SDK 常见情况：
        #
        # content = "..."
        #
        if isinstance(content, str):
            return content.strip()

        # 某些兼容实现可能返回 content list。
        if isinstance(content, list):
            texts: list[str] = []

            for item in content:
                if isinstance(item, dict):
                    text = item.get("text")

                    if text is not None:
                        texts.append(str(text))

                else:
                    text = getattr(item, "text", None)

                    if text is not None:
                        texts.append(str(text))

            return "\n".join(texts).strip()

        return str(content).strip()

    # ============================================================
    # JSON 解析
    # ============================================================

    @staticmethod
    def _parse_json(
        text: str,
    ) -> dict[str, Any] | None:
        """
        尝试把 Qwen-VL 返回的文本解析成 JSON。

        支持：
        1. 纯 JSON
        2. ```json ... ```
        3. JSON 前后存在少量额外文字
        """

        text = text.strip()

        if not text:
            return None

        # --------------------------------------------------------
        # 候选文本
        # --------------------------------------------------------

        candidates = [text]

        # Markdown JSON
        fenced = re.findall(
            r"```(?:json)?\s*(.*?)\s*```",
            text,
            flags=re.S | re.I,
        )

        candidates.extend(fenced)

        # --------------------------------------------------------
        # 逐个尝试
        # --------------------------------------------------------

        for candidate in candidates:
            candidate = candidate.strip()

            if not candidate:
                continue

            try:
                value = json.loads(candidate)

                if isinstance(value, dict):
                    return value

            except json.JSONDecodeError:
                pass

        # --------------------------------------------------------
        # 尝试截取第一个 { 到最后一个 }
        # --------------------------------------------------------

        start = text.find("{")
        end = text.rfind("}")

        if start >= 0 and end > start:
            candidate = text[start : end + 1]

            try:
                value = json.loads(candidate)

                if isinstance(value, dict):
                    return value

            except json.JSONDecodeError:
                pass

        return None

    # ============================================================
    # 分析单帧
    # ============================================================

    def analyze(
        self,
        frame,
        prompt: str,
    ) -> dict[str, Any]:
        """
        对当前单帧图片进行 Qwen-VL 高级视觉理解。

        注意：
            这个函数本身是同步的。

            ActivityUnderstanding 会在后台线程中调用它，
            因此不会阻塞 main.py 摄像头循环。
        """

        # --------------------------------------------------------
        # 图片编码
        # --------------------------------------------------------

        image_data_url = self._image_to_data_url(frame)

        # --------------------------------------------------------
        # System Prompt
        # --------------------------------------------------------

        system_prompt = (
            "你是一个儿童活动视觉理解器。"
            "你负责根据当前摄像头图片和机器视觉感知信息，"
            "理解儿童正在进行的活动。"
            "你只能根据当前图片和提供的感知信息进行判断。"
            "不要编造图片中不存在的物体、动作或关系。"
            "如果图片无法确认某个结论，请使用 unknown，"
            "不要猜测。"
            "输出必须尽量严格遵守用户要求的 JSON 格式。"
        )

        # --------------------------------------------------------
        # User Message
        # --------------------------------------------------------

        messages = [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_data_url,
                        },
                    },
                    {
                        "type": "text",
                        "text": prompt,
                    },
                ],
            },
        ]

        # --------------------------------------------------------
        # 请求 Qwen-VL
        #
        # 注意：
        # 这里使用的是 OpenAI-compatible Chat API。
        #
        # ActivityUnderstanding 的异步机制不在这里实现，
        # 这里只负责一次完整的模型请求。
        # --------------------------------------------------------

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.1,
            )

        except Exception as exc:
            raise RuntimeError(
                f"Qwen-VL 请求失败: {exc}"
            ) from exc

        # --------------------------------------------------------
        # 提取文本
        # --------------------------------------------------------

        raw_text = self._extract_text(response)

        # --------------------------------------------------------
        # JSON
        # --------------------------------------------------------

        parsed = self._parse_json(raw_text)

        # --------------------------------------------------------
        # 返回统一结构
        #
        # 保持和原来的 ActivityUnderstanding / main.py
        # 数据结构兼容。
        # --------------------------------------------------------

        return {
            "model": self.model,
            "raw_text": raw_text,
            "parsed": parsed,
        }