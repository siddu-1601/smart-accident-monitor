# backend/debug_severity.py
#
# Debug script to run FULL ML inference locally/Colab.
# This is NOT used by the Render deployment.

import argparse
from .severity_inference_ml import predict_severity_ml  # type: ignore


def debug_video(video_path: str) -> None:
    sev, acc_type = predict_severity_ml(video_path)
    # predict_severity_ml already prints probabilities and mapping.
    print(f"[debug] Final decision: severity={sev}, accident_type={acc_type}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("video", help="Path to video file (local or /static/uploads/...)")
    args = parser.parse_args()
    debug_video(args.video)
