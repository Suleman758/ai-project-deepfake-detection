import os
import sys
import cv2
import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
import joblib
from pathlib import Path
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from werkzeug.utils import secure_filename

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import MODELS_DIR
from utils import logger

app = Flask(__name__)
app.secret_key = "deepfake-detector-secret"
app.config["UPLOAD_FOLDER"] = Path(__file__).parent / "uploads"
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024
app.config["ALLOWED_EXTENSIONS"] = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv"}
app.config["UPLOAD_FOLDER"].mkdir(parents=True, exist_ok=True)

# ---------- Models (loaded once at startup) ----------
device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device: {device}")

model_path = MODELS_DIR / "xgboost_deepfake.pkl"
if not model_path.exists():
    logger.error(f"Model not found at {model_path}")
    xgb_model = None
else:
    xgb_model = joblib.load(model_path)
    logger.info("XGBoost loaded")

# OpenCV DNN face detector
prototxt = str(MODELS_DIR / "deploy.prototxt")
caffemodel = str(MODELS_DIR / "res10_300x300_ssd_iter_140000.caffemodel")
face_net = cv2.dnn.readNetFromCaffe(prototxt, caffemodel)
logger.info("Face detector loaded")

# ResNet50 feature extractor
resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
resnet = nn.Sequential(*list(resnet.children())[:-1])
resnet.eval()
resnet.to(device)
logger.info("ResNet50 loaded")


def detect_face(frame: np.ndarray) -> np.ndarray | None:
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
    face_net.setInput(blob)
    detections = face_net.forward()

    best_conf, best_box = 0.0, None
    for i in range(detections.shape[2]):
        conf = detections[0, 0, i, 2]
        if conf > best_conf:
            best_box = (detections[0, 0, i, 3:7] * np.array([w, h, w, h])).astype(int)
            best_conf = conf

    if best_box is None or best_conf < 0.5:
        return None
    x1, y1, x2, y2 = best_box[0], best_box[1], best_box[2], best_box[3]
    x1, y1 = max(0, x1), max(0, y1)
    face = frame[y1:y2, x1:x2]
    return face if face.size > 0 else None


@torch.no_grad()
def extract_features(images: list) -> np.ndarray:
    if not images:
        return np.empty((0, 2048))
    batch = np.stack(images).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    batch = (batch - mean) / std
    tensor = torch.from_numpy(batch).permute(0, 3, 1, 2).to(device)
    feats = resnet(tensor)
    return feats.view(feats.size(0), -1).cpu().numpy()


def process_video(video_path: Path) -> dict:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {"error": "Cannot open video file"}

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps if fps > 0 else 0

    num_samples = min(25, total_frames)
    indices = np.linspace(0, total_frames - 1, num_samples, dtype=int)

    face_images = []
    frame_results = []

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret:
            continue

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face = detect_face(frame_rgb)

        if face is not None:
            face = cv2.resize(face, (224, 224))
            face_images.append(face)
            frame_results.append({"frame": int(idx), "face_detected": True})
        else:
            frame_results.append({"frame": int(idx), "face_detected": False})

    cap.release()

    if not face_images:
        return {
            "error": "No faces detected in any frame",
            "total_frames": total_frames,
            "duration": round(duration, 1),
            "frames_checked": len(indices),
            "frame_results": frame_results,
        }

    features = extract_features(face_images)
    frame_probs = xgb_model.predict_proba(features)[:, 1]
    frame_preds = (frame_probs >= 0.5).astype(int)

    video_prob = float(np.mean(frame_probs))
    video_pred = int(video_prob >= 0.5)
    label = "FAKE" if video_pred == 1 else "REAL"

    face_idx = 0
    for fr in frame_results:
        if fr["face_detected"]:
            fr["probability"] = round(float(frame_probs[face_idx]), 4)
            fr["prediction"] = "FAKE" if frame_preds[face_idx] == 1 else "REAL"
            face_idx += 1

    return {
        "label": label,
        "confidence": round(video_prob, 4),
        "total_frames": total_frames,
        "duration": round(duration, 1),
        "fps": round(fps, 1),
        "faces_detected": len(face_images),
        "frames_checked": len(indices),
        "frame_results": frame_results,
        "error": None,
    }


def allowed_file(filename: str) -> bool:
    return Path(filename).suffix.lower() in app.config["ALLOWED_EXTENSIONS"]


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "video" not in request.files:
            flash("No file selected")
            return redirect(request.url)

        file = request.files["video"]
        if file.filename == "":
            flash("No file selected")
            return redirect(request.url)
        if not allowed_file(file.filename):
            flash("Unsupported format. Use: mp4, avi, mov, mkv, webm, flv")
            return redirect(request.url)
        if xgb_model is None:
            flash("Model not found. Run training first!")
            return redirect(request.url)

        filename = secure_filename(file.filename)
        video_path = app.config["UPLOAD_FOLDER"] / filename
        file.save(video_path)
        result = process_video(video_path)
        try:
            video_path.unlink()
        except OSError:
            pass

        if result.get("error"):
            flash(result["error"])
            return render_template("index.html", result=result)
        return render_template("index.html", result=result)

    return render_template("index.html")


@app.route("/api/predict", methods=["POST"])
def api_predict():
    if "video" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["video"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400
    if not allowed_file(file.filename):
        return jsonify({"error": "Unsupported format"}), 400
    if xgb_model is None:
        return jsonify({"error": "Model not found. Run training first!"}), 500

    filename = secure_filename(file.filename)
    video_path = app.config["UPLOAD_FOLDER"] / filename
    file.save(video_path)
    result = process_video(video_path)
    try:
        video_path.unlink()
    except OSError:
        pass

    response = jsonify(result)
    response.headers.add("Access-Control-Allow-Origin", "*")
    return response


if __name__ == "__main__":
    logger.info("Starting Deepfake Detector on http://127.0.0.1:5000")
    app.run(debug=True, host="127.0.0.1", port=5000)
