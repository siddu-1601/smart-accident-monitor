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
        ['major', 'minor', 'severe'] -> indices 0, 1, 2

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

    # Load weights from checkpoint (no backbone.* prefix)
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
    Core entrypoint used by the FastAPI background worker.

    Flow:
    1. Resolve video path on disk.
    2. Sample frames.
    3. Run model on frames as a batch.
    4. Aggregate probabilities across frames.
       - MAX aggregation (worst frame wins).
    5. Apply rule-based thresholds on top of model outputs.
    6. Map to severity class.

    Returns:
        severity: "MINOR" | "MAJOR" | "SEVERE"
        accident_type: currently constant "generic"
    """
    model, class_to_idx = _load_model()

    # Make sure expected labels exist
    if not {"minor", "major", "severe"}.issubset(class_to_idx.keys()):
        raise RuntimeError(f"class_to_idx missing expected keys: {class_to_idx}")

    idx_minor = class_to_idx["minor"]
    idx_major = class_to_idx["major"]
    idx_severe = class_to_idx["severe"]

    real_path = _resolve_video_path(video_path)
    frames = _extract_frames(real_path)  # (N, C, H, W)
    frames = frames.to(device)

    with torch.no_grad():
        outputs = model(frames)                # (N, num_classes)
        probs = torch.softmax(outputs, dim=1)  # (N, num_classes)

        # MAX aggregation across time: treat severity as the worst observed frame
        max_probs, _ = probs.max(dim=0)        # (num_classes,)

    # Align probabilities to semantic labels using class_to_idx
    p_minor = float(max_probs[idx_minor].item())
    p_major = float(max_probs[idx_major].item())
    p_severe = float(max_probs[idx_severe].item())

    # -----------------------------------------------------------------------
    # Rule-based escalation logic on top of the model
    # -----------------------------------------------------------------------
    # These thresholds are chosen for safety bias: we prefer to over-call
    # MAJOR/SEVERE rather than under-report a fatal accident.
    #
    # You can tune these if logs show different behaviour.
    # -----------------------------------------------------------------------

    # Severe if SEVERE prob is clearly high
    if p_severe >= 0.60:
        severity = "SEVERE"
    # Major if MAJOR is strong, or SEVERE is moderate
    elif p_major >= 0.50 or p_severe >= 0.40:
        severity = "MAJOR"
    # Otherwise treat as minor
    else:
        severity = "MINOR"

    # Optional: log probabilities for debugging in Render logs
    print(
        f"[severity] video={video_path} "
        f"p_minor={p_minor:.3f} p_major={p_major:.3f} p_severe={p_severe:.3f} "
        f"-> {severity}"
    )

    accident_type = "generic"
    return severity, accident_type


# ---------------------------------------------------------------------------
# Local CLI test hook (optional)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run severity prediction on a video file")
    parser.add_argument("video", help="Path to video file")
    args = parser.parse_args()

    sev, acc_type = predict_severity(args.video)
    print("Severity:", sev)
    print("Accident type:", acc_type)
