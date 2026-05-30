import numpy as np
from pathlib import Path
import joblib
import argparse

from config import MODELS_DIR, FRAME_SAMPLE_RATE, MAX_FRAMES_PER_VIDEO, TARGET_FRAME_SIZE
from utils import extract_frames, detect_faces, logger
from feature_extractor import ResNetFeatureExtractor


def predict_video(
    video_path: Path,
    model_path: Path = MODELS_DIR / "xgboost_deepfake.pkl",
    aggregation: str = "average",
) -> dict:
    """
    Run deepfake detection on a single video.

    Args:
        video_path: Path to video file.
        model_path: Path to saved XGBoost model.
        aggregation: 'average' (default), 'majority', or 'max' for frame-level fusion.

    Returns:
        dict with video name, frame predictions, video-level prediction, and confidence.
    """
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}. Train first!")

    # Load model
    model: joblib.Memory = joblib.load(model_path)
    extractor = ResNetFeatureExtractor()

    # Extract frames and faces
    logger.info(f"Processing {video_path.name} ...")
    frames = extract_frames(video_path, sample_rate=FRAME_SAMPLE_RATE, max_frames=MAX_FRAMES_PER_VIDEO)
    if not frames:
        return {"video": video_path.name, "error": "No frames extracted"}

    faces = detect_faces(frames)
    valid_count = sum(1 for f in faces if f is not None)
    if valid_count == 0:
        return {"video": video_path.name, "error": "No faces detected"}

    # Extract features
    features = extractor.extract(faces)
    if features.size == 0:
        return {"video": video_path.name, "error": "Feature extraction failed"}

    # Predict per frame
    frame_probs = model.predict_proba(features)[:, 1]
    frame_preds = (frame_probs >= 0.5).astype(int)

    # Video-level aggregation
    if aggregation == "average":
        video_prob = float(np.mean(frame_probs))
        video_pred = int(video_prob >= 0.5)
    elif aggregation == "majority":
        video_pred = int(np.mean(frame_preds) >= 0.5)
        video_prob = float(np.mean(frame_probs))
    elif aggregation == "max":
        video_prob = float(np.max(frame_probs))
        video_pred = int(video_prob >= 0.5)
    else:
        raise ValueError(f"Unknown aggregation: {aggregation}")

    label = "FAKE" if video_pred == 1 else "REAL"

    logger.info(
        f"{video_path.name} -> {label} (confidence: {video_prob:.4f}, "
        f"faces: {valid_count}/{len(frames)} frames)"
    )

    return {
        "video": video_path.name,
        "video_prediction": video_pred,
        "video_label": label,
        "video_confidence": video_prob,
        "frame_probs": frame_probs.tolist(),
        "frame_preds": frame_preds.tolist(),
        "faces_detected": valid_count,
        "frames_processed": len(frames),
        "aggregation": aggregation,
    }


def main():
    parser = argparse.ArgumentParser(description="Deepfake video detector")
    parser.add_argument("video", type=str, help="Path to video file")
    parser.add_argument("--model", type=str, default=str(MODELS_DIR / "xgboost_deepfake.pkl"), help="Path to model")
    parser.add_argument("--aggregation", type=str, default="average", choices=["average", "majority", "max"], help="Frame aggregation method")
    args = parser.parse_args()

    result = predict_video(Path(args.video), Path(args.model), args.aggregation)
    if "error" in result:
        logger.error(f"Prediction failed: {result['error']}")
    else:
        print(f"\n{'='*50}")
        print(f"Video:     {result['video']}")
        print(f"Prediction: {result['video_label']}")
        print(f"Confidence: {result['video_confidence']:.4f}")
        print(f"Frames OK:  {result['faces_detected']}/{result['frames_processed']}")
        print(f"Aggregation: {result['aggregation']}")
        print(f"{'='*50}")


if __name__ == "__main__":
    main()
