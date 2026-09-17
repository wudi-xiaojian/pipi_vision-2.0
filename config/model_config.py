# ============================================================
# Activity
# ============================================================

ACTIVITY_CONFIG_PATH = (
    "activities/configs/paper_cup.yaml"
)

ACTIVITY_DISPLAY_MIN_CONFIDENCE = 0.01
ACTIVITY_RATE_HZ = 10.0
ACTIVITY_WINDOW_SECONDS = 2.0


# ============================================================
# Camera
# ============================================================

CAMERA_INDEX = 0
CAMERA_FALLBACK_FPS = 30.0


# ============================================================
# Hand Detector
# ============================================================

HAND_MODEL_PATH = "models/hand_landmarker.task"

HAND_NUM_HANDS = 2
HAND_MIN_DETECTION_CONFIDENCE = 0.5
HAND_MIN_PRESENCE_CONFIDENCE = 0.5
HAND_MIN_TRACKING_CONFIDENCE = 0.5


# ============================================================
# Tracker
#
# 注意：
# Tracker 的活动相关参数已经迁移到 Activity YAML。
# 这里不再保留 Tracker 全局默认值。
# ============================================================


# ============================================================
# VLM
# ============================================================

VLM_MODEL = "qwen3.8-flash"

VLM_BASE_URL = (
    "https://ws-kotxcen8ue9z794f.cn-beijing.maas.aliyuncs.com"
    "/compatible-mode/v1"
)

VLM_API_KEY_ENV = "DASHSCOPE_API_KEY"
VLM_TIMEOUT = 60.0

VLM_COOLDOWN = 1.5

VLM_TRIGGER_EVENTS = {
    "PICK_UP",
    "PLACE",
    "OBJECT_STOPPED",
}