# Deepfake Video Detector

ResNet50 + XGBoost deepfake detection for videos, with a Flask API + React frontend.

---

## Requirements (Fresh Machine)

| Dependency | Version | Purpose |
|------------|---------|---------|
| Python | 3.10+ | Backend / inference |
| Node.js | 20+ | Frontend dev server |
| CUDA GPU | Optional (falls back to CPU) | Faster inference |

---

## Setup

### 1. Clone & Enter

```bash
git clone <repo-url>
cd BNU_PROJECT
```

### 2. Python Environment

```bash
python3 -m venv venv
source venv/bin/activate
pip install torch torchvision opencv-python xgboost scikit-learn flask joblib numpy pillow matplotlib seaborn tqdm
```

### 3. Download Face Detector Models

```bash
mkdir -p models
wget -O models/deploy.prototxt https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt
wget -O models/res10_300x300_ssd_iter_140000.caffemodel \
  https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel
```

### 4. Frontend Dependencies

```bash
cd frontend
npm install
cd ..
```

---

## Run (Inference Only — No Training Needed)

### Terminal 1 — Start the Flask API

```bash
source venv/bin/activate
python app/app.py
```

Serves on `http://127.0.0.1:5000`

### Terminal 2 — Start the Frontend

```bash
cd frontend
npm run dev
```

Open `http://127.0.0.1:5173` in a browser. Upload a video and get a prediction.

> The dev server proxies `/api/*` to Flask automatically.

---

## API Reference

### `POST /api/predict`

Upload a video file (`multipart/form-data`).

```bash
curl -X POST http://127.0.0.1:5000/api/predict \
  -F "video=@/path/to/video.mp4"
```

**Response:**

```json
{
  "label": "FAKE",
  "confidence": 0.8735,
  "duration": 30.3,
  "fps": 24.0,
  "faces_detected": 5,
  "frames_checked": 25,
  "total_frames": 727,
  "frame_results": [
    {"frame": 0, "face_detected": false},
    {"frame": 30, "face_detected": true, "prediction": "FAKE", "probability": 0.8139}
  ]
}
```

---

## How It Works

```
Input Video
  ↓
OpenCV DNN Face Detector  ←  extracts faces from 25 sampled frames
  ↓
ResNet50 (ImageNet)  ←  2048-dim feature vector per face
  ↓
XGBoost Classifier  ←  trained on FaceForensics++ (400 videos)
  ↓
REAL / FAKE  ←  averaged frame probabilities
```

The model was trained on 200 real + 200 fake FaceForensics++ videos using the **same OpenCV DNN** face detector for both classes, preventing the model from learning extraction artifacts instead of actual deepfake features.

---

## Project Structure

```
.
├── app/                   # Flask backend
│   ├── app.py             # API + prediction logic
│   └── uploads/           # Temp uploads
├── frontend/              # React + Vite + Tailwind
│   └── src/
│       ├── App.tsx        # Dark theme UI
│       └── components/ui/
│           └── shader-animation.tsx   # Three.js background
├── models/                # Inference model (tracked in git)
│   ├── xgboost_deepfake.pkl   ←  trained classifier
│   ├── deploy.prototxt         ←  face detector config
│   └── res10_300x300_ssd_iter_140000.caffemodel  ←  face detector weights
├── extract_all.py         # Training script (not needed for inference)
├── config.py
└── utils.py
```
