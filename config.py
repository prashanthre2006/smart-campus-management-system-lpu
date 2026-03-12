import os

ADMIN_USERNAME = os.getenv("ADMIN_USERNAME", "admin")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "change-me")
QR_SCANNER_TOKEN = os.getenv("QR_SCANNER_TOKEN", "")
PHONEPE_WEBHOOK_TOKEN = os.getenv("PHONEPE_WEBHOOK_TOKEN", "")

PERIODS = [
    "Period 1",
    "Period 2",
    "Period 3",
    "Period 4",
    "Period 5",
    "Period 6",
    "Period 7",
    "Period 8",
]

DATA_DIR = "data"
MODEL_DIR = "models"
DB_PATH = "data/attendance.db"
ATTENDANCE_CSV = "attendance.csv"

# Accuracy tuning
CONFIDENCE_THRESHOLD = 80
CONFIDENCE_STRICT = 78
RECOGNITION_ACCEPT_MAX = 78
REQUIRED_HITS = 2
REQUIRED_ENROLL_IMAGES = 1
BLUR_THRESHOLD = 40.0  # variance of Laplacian
BRIGHTNESS_MIN = 25
BRIGHTNESS_MAX = 235

# Optional deep-learning recognition (requires external model files)
USE_DNN_RECOG = False
EMBEDDING_MODEL_PATH = "models/face_embedder.onnx"
EMBEDDING_DB_PATH = "models/embeddings.json"

# Alerts
ALERTS_ENABLED = os.getenv("ALERTS_ENABLED", "false").lower() in {"1", "true", "yes", "on"}
ALERTS_ON_UNKNOWN = os.getenv("ALERTS_ON_UNKNOWN", "true").lower() in {"1", "true", "yes", "on"}
ALERTS_ON_MARKED = os.getenv("ALERTS_ON_MARKED", "true").lower() in {"1", "true", "yes", "on"}
