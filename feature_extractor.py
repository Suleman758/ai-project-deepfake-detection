import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
from pathlib import Path
from typing import List, Optional
from tqdm import tqdm
import joblib

from config import DEVICE, RESNET_MODEL, RESNET_LAYER, FEATURE_DIM
from utils import logger


class ResNetFeatureExtractor:
    """Extract feature vectors from face images using a pre-trained ResNet."""

    def __init__(self, model_name: str = RESNET_MODEL, layer: str = RESNET_LAYER):
        self.device = torch.device(DEVICE)
        self.model_name = model_name
        self.layer = layer

        # load pre-trained ResNet
        model = getattr(models, model_name)(weights="IMAGENET1K_V1")
        self.model = model.to(self.device)
        self.model.eval()

        # register hook to get intermediate features
        self.features = None
        self._register_hook(layer)

        # pre-processing: ImageNet normalization
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)

        logger.info(f"Loaded {model_name} on {self.device}, extracting from layer '{layer}'")

    def _register_hook(self, layer: str):
        """Attach a forward hook to the target layer."""
        target = self._get_layer(layer)
        if target is None:
            raise ValueError(f"Layer '{layer}' not found in {self.model_name}")

        def hook(module, inp, out):
            # average pool -> 1x1 feature vector
            self.features = out.mean(dim=[2, 3]).detach().cpu().numpy()

        target.register_forward_hook(hook)

    def _get_layer(self, name: str):
        """Get a named sub-module from the ResNet."""
        if name == "avgpool":
            return self.model.avgpool
        if name == "fc":
            return self.model.fc
        parts = name.split(".")
        mod = self.model
        for p in parts:
            try:
                idx = int(p)
                mod = mod[idx]
            except (ValueError, IndexError, TypeError):
                mod = getattr(mod, p, None)
            if mod is None:
                return None
        return mod

    @torch.no_grad()
    def extract(self, face_images: List[Optional[np.ndarray]]) -> np.ndarray:
        """Extract features from a list of face images (H,W,3 uint8)."""
        valid_frames = [f for f in face_images if f is not None]
        if not valid_frames:
            return np.array([])

        # stack and prepare batch
        batch = np.stack(valid_frames).astype(np.float32) / 255.0
        batch = torch.from_numpy(batch).permute(0, 3, 1, 2).to(self.device)
        batch = (batch - self.mean) / self.std

        self.features = None
        self.model(batch)
        return self.features  # (N, FEATURE_DIM)

    @torch.no_grad()
    def extract_single(self, face_image: np.ndarray) -> np.ndarray:
        """Extract features from a single face image."""
        return self.extract([face_image])

    def save(self, path: Path):
        """Not needed — ResNet weights are frozen pre-trained."""


def extract_and_save_features(
    extractor: ResNetFeatureExtractor,
    video_paths_real: List[Path],
    video_paths_fake: List[Path],
    output_dir: Path,
):
    """Extract features for all videos and save per-video .npy files."""
    from utils import extract_frames, detect_faces

    output_dir.mkdir(parents=True, exist_ok=True)

    for label, paths in [("real", video_paths_real), ("fake", video_paths_fake)]:
        for vpath in tqdm(paths, desc=f"Extracting {label}"):
            frames = extract_frames(vpath)
            faces = detect_faces(frames)
            features = extractor.extract(faces)
            if features.size == 0:
                logger.warning(f"No faces found in {vpath.name}")
                continue

            # save per-frame features and label
            out_path = output_dir / f"{vpath.stem}_{label}.npy"
            np.save(out_path, {
                "features": features,
                "label": 0 if label == "real" else 1,
                "video": vpath.name,
            })
