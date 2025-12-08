# backend/severity_inference_ml.py
#
# Full ResNet18-based severity inference for LOCAL/COLAB use only.
# This file is NOT used by the Render deployment.

import os
from typing import Tuple

import cv2
import torch
import torch.nn as nn
import numpy as np
from torchvision import models, transforms


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

_model: torch.nn.Module | None = None
_class_to_idx: dict[str, int] | None = None


def _infer_class_to_idx_from_checkpoint(checkpoint: dict) -> dict[str, int]:
    if "class_to_idx" in checkpoint and isinstance(checkpoint["class_to_idx"], dict):
        return checkpoint["class_to_idx"]

    # Your dataset folders: dataset/minor, dataset/major, dataset/severe
    # ImageFolder sorts alphabetically -> ['major', 'minor', 'severe']
    return {"major": 0, "minor": 1, "severe": 2}


def _extract_state_dict_and_classmap(ckpt_raw) -> tuple[dict, dict[str, int]]:
    if isinstance(ckpt_raw, dict) and "state_dict" in ckpt_raw:
        state_dict = ckpt_raw["state_dict"]
        class_to_idx = _infer_class_to_idx_from_checkpoint(ckpt_raw)
        return state_dict, class_to_idx

    if isinstance(ckpt_raw, dict):
        state_dict = ckpt_raw
        class_to_idx = _infer_class_to_idx_from_checkpoint({})
        return state_dict, class_to_idx

    state_dict = ckpt_raw
    class_to_idx = _infer_class_to_idx_from_checkpoint({})
    return state_dict, class_to_idx


def _load_model() -> tuple[torch.nn.Module, dict[str, int]]:
    global _model, _class_to_idx

    if _model is not None and _class_to_idx is not None:
        return _model, _class_to_idx

    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"Severity model file not found at {MODEL_PATH}")

    ckpt_raw = torch.load(MODEL_PATH, map_location=device)
    state_dict, class_to_idx = _extract_state_dict_and_classmap(ckpt_raw)

    num_classes = len(class_to_idx)

    model = models.resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    _model = model
    _class_to_idx = class_to_idx
    return _model, _class_to_idx


def _resolve_video_path(video_path: str) -> str:
    # Absolute path
    if os.path.isabs(video_path) and os.path.exists(video_path):
        return video_path

    # Common app case: '/static/uploads/filename'
    if video_path.startswith("/static/"):
        candidate = os.path.join(BASE_DIR, video_path.lstrip("/"))
        if os.path.exists(candidate):
            return candidate

    # 'static/uploads/filename'
    if video_path.startswith("static/"):
        candidate = os.path.join(BASE_DIR, video_path)
        if os.path.exists(candidate):
            return candidate

    # Fallback: relative to project root
    project_root = os.path.dirname(BASE_DIR)
    candidate = os.path.join(project_root, video_path.lstrip("/"))
    return candidate


def _extract_frames(video_fs_path: str, num_frames: int = 32) -> torch.Tensor:
    if not os.path.exists(video_fs_path):
        raise RuntimeError(f"Video file does not exist: {video_fs_path}")

    cap = cv2.VideoCapture(video_fs_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_fs_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

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


def predict_severity_ml(video_path: str) -> Tuple[str, str]:
    """
    Full ML inference:
    - MAX over frames
    - Argmax over classes
    """
    model, class_to_idx = _load_model()
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    real_path = _resolve_video_path(video_path)
    frames = _extract_frames(real_path)
    frames = frames.to(device)

    with torch.no_grad():
        outputs = model(frames)
        probs = torch.softmax(outputs, dim=1)
        max_probs, _ = probs.max(dim=0)

    pred_idx = int(max_probs.argmax().item())
    raw_class_name = idx_to_class.get(pred_idx, "minor").lower()

    severity_map = {
        "minor": "MINOR",
        "major": "MAJOR",
        "severe": "SEVERE",
    }
    severity = severity_map.get(raw_class_name, "MINOR")

    p_minor = float(max_probs[class_to_idx["minor"]].item())
    p_major = float(max_probs[class_to_idx["major"]].item())
    p_severe = float(max_probs[class_to_idx["severe"]].item())
    print(
        f"[severity-ml] video={video_path} resolved={real_path} "
        f"class_to_idx={class_to_idx} "
        f"p_minor={p_minor:.3f} p_major={p_major:.3f} p_severe={p_severe:.3f} "
        f"-> {severity}"
    )

    accident_type = "generic"
    return severity, accident_type


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("video", help="Path to video file")
    args = parser.parse_args()

    sev, acc_type = predict_severity_ml(args.video)
    print("Severity:", sev)
    print("Accident type:", acc_type)
