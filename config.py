VIDEO_PATH = "Video/video4.mp4"

# Пороги для анализаторов
CROSS_THRESHOLD_RATIO = 0.6      # доля ширины плеч для детектора скрещивания
FACE_TOUCH_THRESHOLD_RATIO = 0.35
FACE_TOUCH_MIN_DURATION = 0.5    # секунд
ANXIETY_WINDOW = 30              # секунд
ANXIETY_PATTERN_COUNT = 3

GESTURE_WINDOW = 2.0             # секунд
GESTURE_STIFF_MAX = 30
GESTURE_EXCESSIVE_MIN = 100
GESTURE_REST_GRACE = 4.0         # льготный период в секундах