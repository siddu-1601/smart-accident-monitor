# backend/cctv_ingestor.py
#
# Lightweight CCTV accident detector that:
#  - Connects to a camera (RTSP URL or local video file)
#  - Watches frames and uses simple motion-based heuristic for "accident"
#  - When an event is detected, saves a short clip
#  - Uploads the clip to your existing /upload-incident endpoint
#
# This is meant to run LOCALLY or on a separate machine, NOT inside Render.

import os
import time
import threading
import collections
from datetime import datetime
from typing import Deque

import cv2
import numpy as np
import requests


# =============================================================================
# CONFIGURATION
# =============================================================================

# Your deployed backend URL (no trailing slash)
BACKEND_BASE_URL = "https://smart-accident-monitor.onrender.com"

# Folder where we temporarily store clipped videos before uploading
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CLIPS_DIR = os.path.join(BASE_DIR, "cctv_clips")
os.makedirs(CLIPS_DIR, exist_ok=True)

# List your cameras here.
# For testing, we use FireAcc_1.mp4 as a fake CCTV stream.
CAMERAS = [
    {
        "name": "TestCamera1",
        # For real CCTV: "rtsp://user:pass@ip:port/stream"
        # For testing: absolute or relative path to a local mp4
        "source": os.path.join(BASE_DIR, "..", "FireAcc_1.mp4"),
        "location_lat": 17.3850,  # dummy coords
        "location_lng": 78.4867,
    },
]

# Motion detection parameters (TUNED TO BE SENSITIVE)
FRAME_WIDTH = 640
FRAME_HEIGHT = 360
SAMPLE_FPS = 5                  # How many frames per second to analyze
PRE_EVENT_SECONDS = 3           # Seconds BEFORE event to include in clip
POST_EVENT_SECONDS = 3          # Seconds AFTER event to include in clip

# IMPORTANT: we lower these to make sure FireAcc_1.mp4 triggers
MOTION_THRESHOLD = 5.0          # Lower = more sensitive (was 25.0)
MOTION_WINDOW = 2               # How many consecutive frames over threshold


# =============================================================================
# HELPERS
# =============================================================================

def _resize_frame(frame: np.ndarray) -> np.ndarray:
    return cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))


def compute_frame_diff(prev_gray: np.ndarray, curr_gray: np.ndarray) -> float:
    """
    Simple motion metric: mean absolute difference between consecutive frames.
    """
    diff = cv2.absdiff(prev_gray, curr_gray)
    mean_diff = float(np.mean(diff))
    return mean_diff


def save_clip(frames: Deque[np.ndarray], output_path: str, fps: int) -> None:
    """
    Save a sequence of frames as an mp4 video.
    """
    if not frames:
        raise RuntimeError("No frames to save in clip")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    h, w, _ = frames[0].shape
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    for f in frames:
        writer.write(f)

    writer.release()


def upload_clip_to_backend(
    clip_path: str,
    location_lat: float,
    location_lng: float,
    description: str,
) -> None:
    """
    Call your existing /upload-incident endpoint with the clipped video.
    """
    url = f"{BACKEND_BASE_URL}/upload-incident"

    with open(clip_path, "rb") as f:
        files = {
            "video": (os.path.basename(clip_path), f, "video/mp4"),
        }
        data = {
            "location_lat": str(location_lat),
            "location_lng": str(location_lng),
            "description": description,
        }

        print(f"[upload] Sending clip {clip_path} to {url}")
        resp = requests.post(url, files=files, data=data, timeout=60)

    if resp.status_code in (200, 303):
        print(f"[upload] Incident uploaded successfully: status={resp.status_code}")
    else:
        print(f"[upload] FAILED to upload incident: status={resp.status_code}, body={resp.text}")


# =============================================================================
# MAIN CCTV MONITOR LOGIC
# =============================================================================

def monitor_camera(camera_cfg: dict) -> None:
    name = camera_cfg["name"]
    source = camera_cfg["source"]
    lat = camera_cfg["location_lat"]
    lng = camera_cfg["location_lng"]

    print(f"[cctv] Starting monitor for {name} source={source}")

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[cctv] ERROR: failed to open source for camera {name}: {source}")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, SAMPLE_FPS)

    # How many frames to keep before event
    pre_event_frames = PRE_EVENT_SECONDS * SAMPLE_FPS
    post_event_frames = POST_EVENT_SECONDS * SAMPLE_FPS

    buffer: Deque[np.ndarray] = collections.deque(
        maxlen=pre_event_frames + post_event_frames
    )
    prev_gray = None
    motion_counter = 0
    in_event_cooldown = False
    cooldown_frames_remaining = 0
    frame_index = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print(f"[cctv] End of stream or read error for {name}")
                break

            frame_index += 1
            frame_resized = _resize_frame(frame)
            buffer.append(frame_resized)

            # Convert to gray for motion detection
            gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)

            if prev_gray is not None:
                diff_value = compute_frame_diff(prev_gray, gray)

                # DEBUG: log every 10th frame so you can see motion scores
                if frame_index % 10 == 0:
                    print(
                        f"[cctv] {name} frame={frame_index} diff={diff_value:.2f} "
                        f"motion_counter={motion_counter}"
                    )

                if diff_value > MOTION_THRESHOLD:
                    motion_counter += 1
                else:
                    motion_counter = 0

                # Trigger accident event if enough consecutive high-motion frames
                if not in_event_cooldown and motion_counter >= MOTION_WINDOW:
                    print(
                        f"[cctv] POSSIBLE ACCIDENT detected on {name}, "
                        f"frame={frame_index}, diff={diff_value:.2f}"
                    )

                    # Copy current buffer to save as clip
                    clip_frames = list(buffer)
                    timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
                    clip_filename = f"{timestamp}_{name}_accident.mp4"
                    clip_path = os.path.join(CLIPS_DIR, clip_filename)

                    try:
                        save_clip(collections.deque(clip_frames), clip_path, SAMPLE_FPS)
                        print(f"[cctv] Clip saved: {clip_path}")

                        description = f"Auto-detected accident from camera {name}"
                        upload_clip_to_backend(clip_path, lat, lng, description)
                    except Exception as e:
                        print(f"[cctv] ERROR while saving/uploading clip for {name}: {e}")

                    # Enter cooldown so we don't spam multiple incidents for same event
                    in_event_cooldown = True
                    cooldown_frames_remaining = post_event_frames
                    motion_counter = 0

                if in_event_cooldown:
                    cooldown_frames_remaining -= 1
                    if cooldown_frames_remaining <= 0:
                        in_event_cooldown = False

            prev_gray = gray

            # Sleep to approximate SAMPLE_FPS
            time.sleep(1.0 / SAMPLE_FPS)

    finally:
        cap.release()
        print(f"[cctv] Stopped monitor for {name}")


def main() -> None:
    threads = []

    for cam in CAMERAS:
        t = threading.Thread(target=monitor_camera, args=(cam,), daemon=True)
        t.start()
        threads.append(t)

    print("[cctv] All camera monitor threads started. Press Ctrl+C to stop.")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("[cctv] KeyboardInterrupt, exiting...")


if __name__ == "__main__":
    main()
