# ============================================================
# Activity
# ============================================================

ACTIVITY_CONFIG_PATH = (
    "activities/configs/stacking_coins.yaml"
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
# Object Detector
# ============================================================

OBJECT_MODEL_SIZE = "s"
OBJECT_IMGSZ = 640
OBJECT_DEVICE = "auto"
OBJECT_DUP_IOU = 0.65

PROMPT_OBJECT_CONFIDENCE = 0.02


# ============================================================
# Tracker
# ============================================================

TRACKER_DETECT_EVERY = 1
TRACKER_MATCH_IOU = 0.25
TRACKER_MAX_COAST_FRAMES = 5
TRACKER_MIN_HITS = 3

TRACKER_MERGE_IOU = 0.60
TRACKER_MERGE_CONTAIN = 0.80
TRACKER_MERGE_STREAK = 3

TRACKER_SPEED_HISTORY_SIZE = 5
TRACKER_SPEED_EMA_ALPHA = 0.35
TRACKER_MAX_CENTER_JUMP_RATIO = 3.0
TRACKER_LOW_CONFIDENCE_THRESHOLD = 0.10
TRACKER_COAST_SPEED_DECAY = 0.80


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