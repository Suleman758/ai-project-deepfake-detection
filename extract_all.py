import torch
import torch.nn as nn
import torchvision.models as models
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import joblib
import warnings
warnings.filterwarnings("ignore")

from config import MODELS_DIR, FEATURES_DIR
from utils import logger

device = "cuda" if torch.cuda.is_available() else "cpu"

# ResNet50 feature extractor
resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
resnet = nn.Sequential(*list(resnet.children())[:-1])
resnet.eval()
resnet.to(device)
logger.info(f"ResNet50 loaded on {device}")

IMG_SIZE = 224

# OpenCV DNN face detector (same for both fake & real)
prototxt = str(MODELS_DIR / "deploy.prototxt")
caffemodel = str(MODELS_DIR / "res10_300x300_ssd_iter_140000.caffemodel")
face_net = cv2.dnn.readNetFromCaffe(prototxt, caffemodel)


def preprocess(img: np.ndarray) -> torch.Tensor:
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = (img - mean) / std
    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)
    return img


@torch.no_grad()
def extract_features(images: list) -> np.ndarray:
    if not images:
        return np.empty((0, 2048))
    tensors = torch.cat([preprocess(img) for img in images], dim=0)
    feats = resnet(tensors)
    return feats.view(feats.size(0), -1).cpu().numpy()


def detect_face(frame: np.ndarray) -> np.ndarray | None:
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
    face_net.setInput(blob)
    detections = face_net.forward()

    best_conf, best_box = 0.0, None
    for i in range(detections.shape[2]):
        conf = detections[0, 0, i, 2]
        if conf > best_conf:
            box = (detections[0, 0, i, 3:7] * np.array([w, h, w, h])).astype(int)
            best_box = box
            best_conf = conf

    if best_box is None or best_conf < 0.5:
        return None
    x1, y1, x2, y2 = best_box
    x1, y1 = max(0, x1), max(0, y1)
    face = frame[y1:y2, x1:x2]
    return face if face.size > 0 else None


def extract_faces_from_video(video_path: Path, num_frames: int = 15) -> list:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total < 1:
        cap.release()
        return []
    indices = np.linspace(0, total - 1, min(num_frames, total), dtype=int)
    faces = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face = detect_face(frame)
        if face is not None:
            face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
            faces.append(face)
    cap.release()
    return faces


def main():
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    video_data = []

    ffpp_dir = Path("/home/syed/BNU_PROJECT/FF++")
    exts = {".mp4", ".avi", ".mov", ".mkv"}

    # ---- FAKE: extract from FF++/fake ----
    fake_dir = ffpp_dir / "fake"
    fake_vids = sorted([v for v in fake_dir.glob("*") if v.suffix.lower() in exts])
    logger.info(f"Processing {len(fake_vids)} fake videos...")

    for vpath in tqdm(fake_vids, desc="Fake (OpenCV DNN)"):
        faces = extract_faces_from_video(vpath, num_frames=15)
        if faces:
            feats = extract_features(faces)
            video_data.append({"feats": feats, "label": 1, "video": vpath.stem})

    # ---- REAL: extract from FF++/real (same OpenCV DNN) ----
    real_dir = ffpp_dir / "real"
    real_vids = sorted([v for v in real_dir.glob("*") if v.suffix.lower() in exts])
    logger.info(f"Processing {len(real_vids)} real videos...")

    for vpath in tqdm(real_vids, desc="Real (OpenCV DNN)"):
        faces = extract_faces_from_video(vpath, num_frames=15)
        if faces:
            feats = extract_features(faces)
            video_data.append({"feats": feats, "label": 0, "video": vpath.stem})

    logger.info(f"Total videos: {len(video_data)}")
    logger.info(f"  Fake: {sum(1 for v in video_data if v['label']==1)}")
    logger.info(f"  Real: {sum(1 for v in video_data if v['label']==0)}")

    if len(video_data) < 10:
        logger.error("Too few videos, aborting training")
        return

    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, classification_report, confusion_matrix
    import xgboost as xgb
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    video_labels = np.array([v["label"] for v in video_data])
    video_indices = np.arange(len(video_data))

    train_idx, test_idx = train_test_split(
        video_indices, test_size=0.2, random_state=42, stratify=video_labels
    )
    train_labels = video_labels[train_idx]
    train_idx, val_idx = train_test_split(
        train_idx, test_size=0.125, random_state=42, stratify=train_labels
    )

    X_train_list, y_train_list = [], []
    for i in train_idx:
        v = video_data[i]
        X_train_list.append(v["feats"])
        y_train_list.extend([v["label"]] * v["feats"].shape[0])
    X_train = np.vstack(X_train_list)
    y_train = np.array(y_train_list)

    X_val_list, y_val_list = [], []
    for i in val_idx:
        v = video_data[i]
        X_val_list.append(v["feats"])
        y_val_list.extend([v["label"]] * v["feats"].shape[0])
    X_val = np.vstack(X_val_list)
    y_val = np.array(y_val_list)

    logger.info(f"Train: {len(train_idx)} videos, Val: {len(val_idx)}, Test: {len(test_idx)}")
    logger.info(f"Train frames: {X_train.shape[0]}, Val frames: {X_val.shape[0]}")

    model = xgb.XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        reg_lambda=2.0, reg_alpha=1.0,
        eval_metric="logloss", random_state=42, n_jobs=-1,
        use_label_encoder=False,
    )
    model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_val, y_val)], verbose=50)

    for name, Xs, ys in [("Train", X_train, y_train), ("Val", X_val, y_val)]:
        yp = model.predict(Xs)
        ypr = model.predict_proba(Xs)[:, 1]
        logger.info(f"[Frame] {name:>6} Acc: {accuracy_score(ys, yp):.4f}  AUC: {roc_auc_score(ys, ypr):.4f}  F1: {f1_score(ys, yp):.4f}")

    logger.info("\n--- Video-Level Test ---")
    y_true_vid, y_pred_vid, y_prob_vid = [], [], []
    for i in test_idx:
        v = video_data[i]
        frame_probs = model.predict_proba(v["feats"])[:, 1]
        video_prob = float(np.mean(frame_probs))
        video_pred = int(video_prob >= 0.5)
        y_true_vid.append(v["label"])
        y_pred_vid.append(video_pred)
        y_prob_vid.append(video_prob)

    y_true_vid = np.array(y_true_vid)
    y_pred_vid = np.array(y_pred_vid)
    y_prob_vid = np.array(y_prob_vid)

    acc = accuracy_score(y_true_vid, y_pred_vid)
    auc = roc_auc_score(y_true_vid, y_prob_vid)
    f1 = f1_score(y_true_vid, y_pred_vid)
    logger.info(f"Test  Acc: {acc:.4f}  AUC: {auc:.4f}  F1: {f1:.4f}")
    logger.info("\n" + classification_report(y_true_vid, y_pred_vid, target_names=["Real", "Fake"]))

    cm = confusion_matrix(y_true_vid, y_pred_vid)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Real", "Fake"], yticklabels=["Real", "Fake"])
    plt.title("Confusion Matrix (Video-Level)")
    plt.savefig(str(MODELS_DIR / "confusion_matrix.png"), dpi=150, bbox_inches="tight")

    model_path = str(MODELS_DIR / "xgboost_deepfake.pkl")
    joblib.dump(model, model_path)
    logger.info(f"Model saved to {model_path}")


if __name__ == "__main__":
    main()
