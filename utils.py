import cv2
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# OpenCV DNN face detector (SSD + ResNet)
_face_net = None


def _get_face_net():
    global _face_net
    if _face_net is None:
        prototxt = Path(__file__).parent / "models" / "deploy.prototxt"
        caffemodel = Path(__file__).parent / "models" / "res10_300x300_ssd_iter_140000.caffemodel"
        if not prototxt.exists() or not caffemodel.exists():
            raise RuntimeError("Face detector model files not found in models/. Run setup first.")
        _face_net = cv2.dnn.readNetFromCaffe(str(prototxt), str(caffemodel))
    return _face_net


def extract_frames(
    video_path: Path,
    sample_rate: int = 5,
    max_frames: int = 60,
    target_size: Tuple[int, int] = (224, 224),
) -> List[np.ndarray]:
    """Extract evenly-spaced frames from a video file."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.warning(f"Cannot open video: {video_path}")
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return []

    # sample indices evenly across the video
    step = max(1, sample_rate)
    indices = list(range(0, total_frames, step))[:max_frames]

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, target_size)
        frames.append(frame)

    cap.release()
    return frames


def detect_faces(frames: List[np.ndarray], confidence: float = 0.5) -> List[Optional[np.ndarray]]:
    """Detect and crop the largest face from each frame using OpenCV DNN SSD."""
    net = _get_face_net()

    face_frames = []
    for frame in frames:
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
        net.setInput(blob)
        detections = net.forward()

        best_conf = 0.0
        best_box = None
        for i in range(detections.shape[2]):
            conf = detections[0, 0, i, 2]
            if conf > best_conf and conf >= confidence:
                box = (detections[0, 0, i, 3:7] * np.array([w, h, w, h])).astype(int)
                best_box = box
                best_conf = conf

        if best_box is None:
            face_frames.append(None)
            continue

        x1, y1, x2, y2 = best_box
        x1, y1 = max(0, x1), max(0, y1)
        face = frame[y1:y2, x1:x2]
        if face.size == 0:
            face_frames.append(None)
            continue
        face = cv2.resize(face, (224, 224))
        face_frames.append(face)

    return face_frames


def frame_to_tensor(frame: np.ndarray) -> np.ndarray:
    """Normalize frame: float32 in [0,1]."""
    return frame.astype(np.float32) / 255.0
