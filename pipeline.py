from pathlib import Path
from config import (
    DATA_DIR, FRAMES_DIR, FEATURES_DIR, MODELS_DIR,
    REAL_VIDEOS_DIR, FAKE_VIDEOS_DIR,
)
from utils import logger, extract_frames, detect_faces
from feature_extractor import ResNetFeatureExtractor, extract_and_save_features
import train_xgboost
import numpy as np
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")


def gather_videos():
    real_vids = sorted(REAL_VIDEOS_DIR.glob("*"))
    fake_vids = sorted(FAKE_VIDEOS_DIR.glob("*"))
    # filter to common video extensions
    exts = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v", ".flv"}
    real_vids = [v for v in real_vids if v.suffix.lower() in exts]
    fake_vids = [v for v in fake_vids if v.suffix.lower() in exts]
    logger.info(f"Found {len(real_vids)} real videos, {len(fake_vids)} fake videos")
    assert len(real_vids) > 0 and len(fake_vids) > 0, "Need at least 1 real and 1 fake video"
    return real_vids, fake_vids


def main():
    logger.info("=" * 60)
    logger.info("Deepfake Video Detector — ResNet + XGBoost Pipeline")
    logger.info("=" * 60)

    # Step 1: Gather videos
    real_vids, fake_vids = gather_videos()

    # Step 2: Extract features
    extractor = ResNetFeatureExtractor()
    extract_and_save_features(extractor, real_vids, fake_vids, FEATURES_DIR)

    # Step 3: Train XGBoost
    train_xgboost.main()

    logger.info("Pipeline complete!")


if __name__ == "__main__":
    main()
