# backend/severity_inference.py
#
# FULL ML INFERENCE VERSION FOR RENDER.
# Uses ResNet18 on video frames to predict MINOR/MAJOR/SEVERE.

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
# Global model state
# ---------------------------------------------------------------------------

_model: torch.nn.Module | None = None
_class_to_idx: dict[str, int] | None = None


def _infer_class_to_idx_from_checkpoint(checkpoint: dict) -> dict[str, int]:
    """
    Try to read class_to_idx from checkpoint; if missing, fall back
    to the training-time folder order you used.

    Dataset structure:
        dataset/minor
        dataset/major
        dataset/severe

    ImageFolder sorts class names alphabetically:
        ['major', 'minor', 'severe'] -> indices 0,1,2

    So default mapping = {"major": 0, "minor": 1, "severe": 2}
    """
    if "class_to_idx" in checkpoint and isinstance(checkpoint["class_to_idx"], dict):
        return checkpoint["class_to_idx"]

    return {"major": 0, "minor": 1, "severe": 2}


def _extract_state_dict_and_classmap(ckpt_raw) -> tuple[dict, dict[str, int]]:
    """
    Handle different checkpoint formats:
    - torch.save(model.state_dict())
    - torch.save({"state_dict": ..., "class_to_idx": ...})
    """
    if isinstance(ckpt_raw, dict) and "state_dict" in ckpt_raw:
        state_dict = ckpt_raw["state_dict"]
        class_to_idx = _infer_class_to_idx_from_checkpoint(ckpt_raw)
        return state_dict, class_to_idx

    if isinstance(ckpt_raw, dict):
        # Could be a plain state_dict saved via torch.save(model.state_dict())
        state_dict = ckpt_raw
        class_to_idx = _infer_class_to_idx_from_checkpoint({})
        return state_dict, class_to_idx

    # Fallback: treat as raw state_dict
    state_dict = ckpt_raw
    class_to_idx = _infer_class_to_idx_from_checkpoint({})
    return state_dict, class_to_idx


def _load_model() -> tuple[torch.nn.Module, dict[str, int]]:
    """
    Lazy-load model + class mapping once per process.

    Uses a plain ResNet18 so keys like "conv1.weight", "layer1.0.conv1.weight",
    "fc.weight" match your trained checkpoint.
    """
    global _model, _class_to_idx

    if _model is not None and _class_to_idx is not None:
        return _model, _class_to_idx

    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"Severity model file not found at {MODEL_PATH}")

    ckpt_raw = torch.load(MODEL_PATH, map_location=device)
    state_dict, class_to_idx = _extract_state_dict_and_classmap(ckpt_raw)

    num_classes = len(class_to_idx)

    # Build plain ResNet18, then swap final FC
    model = models.resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    _model = model
    _class_to_idx = class_to_idx  # {"major":0,"minor":1,"severe":2}
    return _model, _class_to_idx


# ---------------------------------------------------------------------------
# Video utilities
# ---------------------------------------------------------------------------

def _resolve_video_path(video_path: str) -> str:
    """
    Convert DB-stored path (e.g. '/static/uploads/xyz.mp4')
    into an actual filesystem path on the server.

    Files are saved to backend/static/uploads/...
    DB stores '/static/uploads/filename.mp4'
    BASE_DIR == backend/, so path = BASE_DIR/static/uploads/filename.mp4
    """
    # If it's already an absolute existing path, just use it
    if os.path.isabs(video_path) and os.path.exists(video_path):
        return video_path

    # Normal case in this app: '/static/uploads/filename'
    if video_path.startswith("/static/"):
        candidate = os.path.join(BASE_DIR, video_path.lstrip("/"))
        if os.path.exists(candidate):
            return candidate

    # Also support 'static/uploads/filename' (no leading slash)
    if video_path.startswith("static/"):
        candidate = os.path.join(BASE_DIR, video_path)
        if os.path.exists(candidate):
            return candidate

    # Fallback: relative to project root (parent of backend/)
    project_root = os.path.dirname(BASE_DIR)
    candidate = os.path.join(project_root, video_path.lstrip("/"))
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

    # If metadata is broken, fall back to sequential reads
    if total_frames <= 0:
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

    # Uniform sampling over the video
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
    FULL ML PIPELINE:

    1. Resolve video path on disk.
    2. Sample frames.
    3. Run model on frames as a batch.
    4. Aggregate per-class probabilities across frames with MAX (worst frame).
    5. Let the model decide via argmax.

    Returns:
        severity: "MINOR" | "MAJOR" | "SEVERE"
        accident_type: currently constant "generic"
    """
    model, class_to_idx = _load_model()

    # Inverse mapping for readability: {idx: "minor"/"major"/"severe"}
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    real_path = _resolve_video_path(video_path)
    frames = _extract_frames(real_path)  # (N, C, H, W)
    frames = frames.to(device)

    with torch.no_grad():
        outputs = model(frames)                # (N, num_classes)
        probs = torch.softmax(outputs, dim=1)  # (N, num_classes)
        max_probs, _ = probs.max(dim=0)        # (num_classes,)

    # Raw model decision
    pred_idx = int(max_probs.argmax().item())
    raw_class_name = idx_to_class.get(pred_idx, "minor").lower()

    severity_map = {
        "minor": "MINOR",
        "major": "MAJOR",
        "severe": "SEVERE",
    }
    severity = severity_map.get(raw_class_name, "MINOR")

    # Log for debugging
    p_minor = float(max_probs[class_to_idx["minor"]].item())
    p_major = float(max_probs[class_to_idx["major"]].item())
    p_severe = float(max_probs[class_to_idx["severe"]].item())
    print(
        f"[severity] video={video_path} resolved={real_path} "
        f"class_to_idx={class_to_idx} "
        f"p_minor={p_minor:.3f} p_major={p_major:.3f} p_severe={p_severe:.3f} "
        f"-> {severity}"
    )

    accident_type = "generic"
    return severity, accident_type


# ---------------------------------------------------------------------------
# Local CLI test hook
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run severity prediction on a video file")
    parser.add_argument("video", help="Path to video file")
    args = parser.parse_args()

    sev, acc_type = predict_severity(args.video)
    print("Severity:", sev)
    print("Accident type:", acc_type)
