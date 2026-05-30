import numpy as np
import joblib
from pathlib import Path
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
)
import xgboost as xgb

from config import FEATURES_DIR, MODELS_DIR, XGB_PARAMS, TEST_SPLIT, VAL_SPLIT
from utils import logger


def load_features_by_video(features_dir: Path):
    """Load features grouped by video. Returns dict of {video_name: {'feats': ndarray, 'label': int}}."""
    video_data = {}
    for fpath in sorted(features_dir.glob("*.npy")):
        data = np.load(fpath, allow_pickle=True).item()
        video_data[data["video"]] = {
            "feats": data["features"],
            "label": data["label"],
        }
    logger.info(f"Loaded {len(video_data)} videos")
    return video_data


def aggregate_video_prediction(frame_probs: np.ndarray, threshold: float = 0.5) -> tuple:
    """Aggregate frame probabilities to a video-level prediction (average)."""
    video_prob = float(np.mean(frame_probs))
    video_pred = int(video_prob >= threshold)
    return video_pred, video_prob


def main():
    logger.info("Loading extracted features...")
    video_data = load_features_by_video(FEATURES_DIR)

    video_names = list(video_data.keys())
    labels = np.array([video_data[v]["label"] for v in video_names])

    # --- Video-level split ---
    v_train, v_test = train_test_split(
        video_names, test_size=TEST_SPLIT, stratify=labels, random_state=42
    )
    train_labels = np.array([video_data[v]["label"] for v in v_train])
    val_frac = VAL_SPLIT / (1 - TEST_SPLIT)
    v_train, v_val = train_test_split(
        v_train, test_size=val_frac, stratify=train_labels, random_state=42
    )
    logger.info(f"Videos — Train: {len(v_train)}, Val: {len(v_val)}, Test: {len(v_test)}")

    # --- Build frame-level training set ---
    X_train_list, y_train_list = [], []
    for v in v_train:
        d = video_data[v]
        X_train_list.append(d["feats"])
        y_train_list.extend([d["label"]] * d["feats"].shape[0])
    X_train = np.vstack(X_train_list)
    y_train = np.array(y_train_list)

    X_val_list, y_val_list = [], []
    for v in v_val:
        d = video_data[v]
        X_val_list.append(d["feats"])
        y_val_list.extend([d["label"]] * d["feats"].shape[0])
    X_val = np.vstack(X_val_list) if X_val_list else np.empty((0, X_train.shape[1]))
    y_val = np.array(y_val_list) if y_val_list else np.array([])

    logger.info(f"Frames — Train: {X_train.shape[0]}, Val: {X_val.shape[0]}")

    # --- Train XGBoost ---
    logger.info("Training XGBoost classifier...")
    model = xgb.XGBClassifier(**XGB_PARAMS)
    eval_set = [(X_train, y_train)]
    if X_val.shape[0] > 0:
        eval_set.append((X_val, y_val))
    model.fit(X_train, y_train, eval_set=eval_set, verbose=100)

    # --- Per-frame evaluation ---
    for split_name, X_split, y_split in [
        ("Train", X_train, y_train),
        ("Val", X_val, y_val),
    ]:
        if X_split.shape[0] == 0:
            continue
        y_pred = model.predict(X_split)
        y_prob = model.predict_proba(X_split)[:, 1]
        logger.info(
            f"[Frame-level] {split_name:>6}  Acc: {accuracy_score(y_split, y_pred):.4f}  "
            f"AUC: {roc_auc_score(y_split, y_prob):.4f}  "
            f"F1: {f1_score(y_split, y_pred):.4f}"
        )

    # --- Video-level evaluation on test set ---
    logger.info("\n--- Video-level Test Set Evaluation ---")
    y_true_video, y_pred_video, y_prob_video = [], [], []
    for v in v_test:
        d = video_data[v]
        frame_probs = model.predict_proba(d["feats"])[:, 1]
        pred, prob = aggregate_video_prediction(frame_probs)
        y_true_video.append(d["label"])
        y_pred_video.append(pred)
        y_prob_video.append(prob)

    y_true_video = np.array(y_true_video)
    y_pred_video = np.array(y_pred_video)
    y_prob_video = np.array(y_prob_video)

    acc = accuracy_score(y_true_video, y_pred_video)
    auc = roc_auc_score(y_true_video, y_prob_video)
    prec = precision_score(y_true_video, y_pred_video)
    rec = recall_score(y_true_video, y_pred_video)
    f1 = f1_score(y_true_video, y_pred_video)

    logger.info(
        f"[Video-level] Test  Acc: {acc:.4f}  AUC: {auc:.4f}  "
        f"Prec: {prec:.4f}  Rec: {rec:.4f}  F1: {f1:.4f}"
    )
    logger.info("\n" + classification_report(y_true_video, y_pred_video, target_names=["Real", "Fake"]))

    # --- Confusion matrix ---
    cm = confusion_matrix(y_true_video, y_pred_video)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Real", "Fake"], yticklabels=["Real", "Fake"])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix — XGBoost Deepfake Detector (Video-Level)")
    plt.savefig(str(MODELS_DIR / "confusion_matrix.png"), dpi=150, bbox_inches="tight")
    logger.info(f"Confusion matrix saved to {MODELS_DIR / 'confusion_matrix.png'}")

    # --- Feature importance ---
    plt.figure(figsize=(10, 6))
    xgb.plot_importance(model, max_num_features=20, height=0.8)
    plt.title("Top 20 Feature Importances")
    plt.savefig(str(MODELS_DIR / "feature_importance.png"), dpi=150, bbox_inches="tight")
    logger.info(f"Feature importance saved to {MODELS_DIR / 'feature_importance.png'}")

    # --- Save model ---
    model_path = MODELS_DIR / "xgboost_deepfake.pkl"
    joblib.dump(model, model_path)
    logger.info(f"Model saved to {model_path}")


if __name__ == "__main__":
    main()
