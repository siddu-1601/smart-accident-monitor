# backend/severity_inference.py

import os
from typing import Tuple


def _resolve_video_path(video_path: str) -> str:
    """
    Keep this helper so debug tools or future versions can still resolve paths,
    but on the Render deployment we DO NOT run heavy ML.
    """
    # Absolute path
    if os.path.isabs(video_path) and os.path.exists(video_path):
        return video_path

    # Normal case: '/static/uploads/filename'
    # We resolve relative to backend/ so that it matches where uploads are saved.
    base_dir = os.path.dirname(os.path.abspath(__file__))
    if video_path.startswith("/"):
        candidate = os.path.join(base_dir, video_path.lstrip("/"))
        return candidate

    # Fallback: treat as relative to project root
    project_root = os.path.dirname(base_dir)
    candidate = os.path.join(project_root, video_path)
    return candidate


def predict_severity(video_path: str) -> Tuple[str, str]:
    """
    LIGHTWEIGHT STUB FOR RENDER DEPLOYMENT.

    This version does NOT load any ML model. It simply:
    - Resolves the path (for logging / debugging if needed)
    - Returns a fixed severity label and generic accident_type.

    The REAL ML inference lives on the 'ml-full' branch and in Colab /
    local scripts, not on the Render instance (to avoid memory/runtime issues).
    """
    resolved = _resolve_video_path(video_path)
    print(f"[severity-stub] Called predict_severity for video={video_path}, resolved={resolved}")

    # For hosted demo, we just treat everything as MINOR or keep some trivial rule.
    # You can change this to "MAJOR" if you want a bit more drama in the UI.
    severity = "MINOR"
    accident_type = "generic"

    return severity, accident_type
