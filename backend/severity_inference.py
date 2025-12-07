# backend/severity_inference.py

import os
from typing import Tuple

import cv2
import torch
import torch.nn as nn
import numpy as np
from torchvision import models, transforms


# ---------------------------------------------------------------------------
# Paths & device
# ---------------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "ml_models", "severity_model.pth")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMG_SIZE = 224

_transform = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)


# ---------------------------------------------------------------------------
# Model definition (must match the training architecture)
# ---------------------------------------------------------------------------

class SeverityModel(nn.Module):
    def __init__(self, num_classes: int = 3):
        super().__init__()
        base = models.resnet18(weights=None)
        in_features = base.fc.in_features
        base.fc = nn.Linear(in_features, num_classes)
        self.backbone = base

    def forward(self, x):
        # x: (B, C, H, W)
        return self.backbone(x)


_model: SeverityModel | None = None
_idx_to_class: dict[int, str] | None = None


def _load_model() -> tuple[SeverityModel, dict[int, str]]:
    """
    Lazy-load model + class mapping once per process.
    """
    global _model, _idx_to_class

    if _model is not None and _idx_to_class is not None:
        return _model, _idx_to_class

    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"Severity model file not found at {MODEL_PATH}")

    checkpoint = torch.load(MODEL_PATH, map_location=device)

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    class_to_idx = checkpoint.get(
        "class_to_idx",
        {"major": 0, "minor": 1, "severe": 2},
    )
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    num_classes = len(idx_to_class)
    model = SeverityModel(num_classes=num_classes)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    _model = model
    _idx_to_class = idx_to_class
    return _model, _idx_to_class


# ---------------------------------------------------------------------------
# Video utilities
# ---------------------------------------------------------------------------

def _resolve_video_path(video_path: str) -> str:
    """
    Convert DB-stored path (e.g. '/static/uploads/xyz.mp4')
    into an actual filesystem path on the server.
    """
    # Absolute path on disk
    if os.path.isabs(video_path) and os.path.exists(video_path):
        return video_path

    # Common case: stored as '/static/uploads/filename'
    if video_path.startswith("/"):
        project_root = os.path.dirname(BASE_DIR)  # parent of backend/
        candidate = os.path.join(project_root, video_path.lstrip("/"))
        if os.path.exists(candidate):
            return candidate

    # Fallback: treat as relative to project root
    project_root = os.path.dirname(BASE_DIR)
    candidate = os.path.join(project_root, video_path)
    return candidate


def _extract_frames(video_fs_path: str, num_frames: int = 16) -> torch.Tensor:
    """
    Read a video and sample up to `num_frames` frames uniformly.
    Returns a tensor of shape (N, C, H, W).
    """
    if not os.path.exists(video_fs_path):
        raise RuntimeError(f"Video file does not exist: {video_fs_path}")

    cap = cv2.VideoCapture(video_fs_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_fs_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        # try to just read sequentially up to some max
        frames = []
        for _ in range(num_frames):
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(_transform(frame))
        cap.release()
        if not frames:
            raise RuntimeError(f"No frames extracted from: {video_fs_path}")
        while len(frames) < num_frames:
            frames.append(frames[-1].clone())
        return torch.stack(frames, dim=0)

    indices = np.linspace(0, total_frames - 1, num_frames).astype(int)
    frames = []

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(_transform(frame))

    cap.release()

    if not frames:
        raise RuntimeError(f"No frames extracted from: {video_fs_path}")

    while len(frames) < num_frames:
        frames.append(frames[-1].clone())

    return torch.stack(frames, dim=0)


# ---------------------------------------------------------------------------
# Public API: predict_severity
# ---------------------------------------------------------------------------

def predict_severity(video_path: str) -> Tuple[str, str]:
    """
    Core entrypoint used by the FastAPI background worker.

    1. Resolve video path on disk.
    2. Sample frames.
    3. Run model on frames as a batch.
    4. Average probabilities across frames.
    5. Map to severity class.
    """
    model, idx_to_class = _load_model()

    real_path = _resolve_video_path(video_path)
    frames = _extract_frames(real_path)  # (N, C, H, W)
    frames = frames.to(device)

    with torch.no_grad():
        outputs = model(frames)  # (N, num_classes)
        probs = torch.softmax(outputs, dim=1)  # (N, num_classes)
        mean_probs = probs.mean(dim=0)  # (num_classes,)
        pred_idx = int(torch.argmax(mean_probs).item())

    class_name = idx_to_class.get(pred_idx, "minor").lower()

    severity_map = {
        "minor": "MINOR",
        "major": "MAJOR",
        "severe": "SEVERE",
    }
    severity = severity_map.get(class_name, "MINOR")

    # You can expand this later if you want specific accident types.
    accident_type = "generic"

    return severity, accident_type
