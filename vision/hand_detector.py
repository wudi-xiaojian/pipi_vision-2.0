from __future__ import annotations

import os

import cv2
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from .hand_types import HandObservation


class HandDetector:
    """
    MediaPipe Hand Landmarker 封装。

    功能：
    1. 检测最多两只手
    2. 输出 Left / Right
    3. 输出 21 个手部关键点
    4. 输出手部 bbox
    5. 输出 wrist 像素坐标
    6. 转换成统一 HandObservation

    注意：
    ------------------------------------------------------------
    当前 main.py 会先对摄像头画面执行：

        frame = cv2.flip(frame, 1)

    也就是说，MediaPipe 接收到的是镜像后的自拍画面。

    MediaPipe 的 handedness 在这种情况下需要做一次左右校正，
    因此这里将 MediaPipe 返回的：

        Left  -> Right
        Right -> Left

    只修正 hand_id，不修改：
        - landmarks
        - bbox
        - wrist
        - confidence
        - handedness_score

    这样不会影响后面的手部坐标、手杯距离以及 ActivityEngine。
    """

    def __init__(
        self,
        model_path: str = "models/hand_landmarker.task",
        num_hands: int = 2,
        min_hand_detection_confidence: float = 0.5,
        min_hand_presence_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Hand Landmarker model not found: {model_path}"
            )

        base_options = python.BaseOptions(
            model_asset_path=model_path
        )

        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_hands=num_hands,
            min_hand_detection_confidence=(
                min_hand_detection_confidence
            ),
            min_hand_presence_confidence=(
                min_hand_presence_confidence
            ),
            min_tracking_confidence=(
                min_tracking_confidence
            ),
        )

        self.landmarker = (
            vision.HandLandmarker.create_from_options(
                options
            )
        )

    @staticmethod
    def _correct_handedness(
        handedness: str,
    ) -> str:
        """
        修正镜像画面下的左右手标签。

        MediaPipe:
            Left  -> 实际显示为 Right
            Right -> 实际显示为 Left

        因此这里进行一次反转。

        其他未知标签保持原样。
        """

        handedness_lower = handedness.strip().lower()

        if handedness_lower == "left":
            return "Right"

        if handedness_lower == "right":
            return "Left"

        return handedness

    def detect(
        self,
        frame,
        timestamp_ms: int,
    ) -> list[HandObservation]:
        """
        对单帧图像进行手部检测。

        参数：
            frame:
                BGR OpenCV 图像。

            timestamp_ms:
                MediaPipe VIDEO 模式要求的时间戳，
                必须随着视频帧单调递增。

        返回：
            list[HandObservation]
        """

        if frame is None:
            return []

        if len(frame.shape) != 3:
            return []

        height, width = frame.shape[:2]

        # --------------------------------------------------------
        # BGR -> RGB
        # --------------------------------------------------------

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB,
        )

        rgb_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame,
        )

        # --------------------------------------------------------
        # MediaPipe Hand Landmarker
        # --------------------------------------------------------

        result = self.landmarker.detect_for_video(
            rgb_image,
            timestamp_ms,
        )

        observations: list[HandObservation] = []

        if not result.hand_landmarks:
            return observations

        # --------------------------------------------------------
        # 处理每只手
        # --------------------------------------------------------

        for index, hand_landmarks in enumerate(
            result.hand_landmarks
        ):

            if index >= len(result.handedness):
                continue

            handedness_list = result.handedness[index]

            if not handedness_list:
                continue

            handedness = handedness_list[0]

            # ----------------------------------------------------
            # MediaPipe 原始 handedness
            # ----------------------------------------------------

            raw_hand_id = (
                handedness.category_name
                or "Unknown"
            )

            # ----------------------------------------------------
            # 左右手校正
            #
            # 当前 main.py 使用了水平镜像，
            # 因此这里反转 Left / Right。
            # ----------------------------------------------------

            hand_id = self._correct_handedness(
                raw_hand_id
            )

            handedness_score = float(
                handedness.score
                if handedness.score is not None
                else 0.0
            )

            # ----------------------------------------------------
            # 21 个关键点
            # ----------------------------------------------------

            pixel_landmarks: list[list[float]] = []

            xs: list[float] = []
            ys: list[float] = []

            for landmark in hand_landmarks:

                x = float(
                    landmark.x * width
                )

                y = float(
                    landmark.y * height
                )

                z = float(
                    landmark.z
                )

                pixel_landmarks.append(
                    [x, y, z]
                )

                xs.append(x)
                ys.append(y)

            if not pixel_landmarks:
                continue

            # ----------------------------------------------------
            # bbox
            # ----------------------------------------------------

            x1 = max(
                0.0,
                min(xs),
            )

            y1 = max(
                0.0,
                min(ys),
            )

            x2 = min(
                float(width),
                max(xs),
            )

            y2 = min(
                float(height),
                max(ys),
            )

            # ----------------------------------------------------
            # wrist
            #
            # MediaPipe 手部 landmark:
            # 0 = wrist
            # ----------------------------------------------------

            wrist_x = pixel_landmarks[0][0]
            wrist_y = pixel_landmarks[0][1]

            # ----------------------------------------------------
            # 统一 HandObservation
            # ----------------------------------------------------

            observation = HandObservation(
                hand_id=hand_id,
                wrist=(
                    wrist_x,
                    wrist_y,
                ),
                bbox=[
                    x1,
                    y1,
                    x2,
                    y2,
                ],
                landmarks=pixel_landmarks,
                confidence=handedness_score,
                handedness_score=handedness_score,
            )

            observations.append(
                observation
            )

        return observations

    def close(self) -> None:
        """释放 MediaPipe 模型。"""

        if self.landmarker is not None:
            self.landmarker.close()
            self.landmarker = None

    def __enter__(self):
        return self

    def __exit__(
        self,
        exc_type,
        exc_value,
        traceback,
    ):
        self.close()