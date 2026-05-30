import os
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
FRAMES_DIR = DATA_DIR / "frames"
FEATURES_DIR = DATA_DIR / "features"
MODELS_DIR = BASE_DIR / "models"

# Datasets — FaceForensics++ (FF++)
FF_DIR = BASE_DIR / "FF++"
REAL_VIDEOS_DIR = FF_DIR / "real"
FAKE_VIDEOS_DIR = FF_DIR / "fake"

# Frame extraction
FRAME_SAMPLE_RATE = 5          # extract 1 frame every N frames
TARGET_FRAME_SIZE = (224, 224) # ResNet input size
MAX_FRAMES_PER_VIDEO = 60      # cap frames per video

# Face detection
FACE_DETECTION_CONFIDENCE = 0.85

# Feature extraction (ResNet)
RESNET_MODEL = "resnet50"      # or "resnet18", "resnet101"
RESNET_LAYER = "avgpool"       # extract features before FC layer
FEATURE_DIM = 2048             # 2048 for resnet50/101, 512 for resnet18

# XGBoost
XGB_PARAMS = {
    "n_estimators": 500,
    "max_depth": 8,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "eval_metric": "logloss",
    "use_label_encoder": False,
    "random_state": 42,
    "n_jobs": -1,
}

# Train / val split
TEST_SPLIT = 0.2
VAL_SPLIT = 0.1

# Device
DEVICE = "cuda" if __import__("torch").cuda.is_available() else "cpu"

os.makedirs(FRAMES_DIR, exist_ok=True)
os.makedirs(FEATURES_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
