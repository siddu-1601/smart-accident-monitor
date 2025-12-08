import sys
import requests
from backend.severity_inference_ml import predict_severity_ml

# ============================
# CONFIG: UPDATE THESE ONLY
# ============================

FAST2SMS_API_KEY = "REMOVED_FOR_GITHUB"
ALERT_PHONE_NUMBER = "REMOVED_FOR_GITHUB"


# ============================
# SMS ALERT FUNCTION
# ============================

def send_severe_alert(video_name):
    url = "https://www.fast2sms.com/dev/bulkV2"

    message = f"SEVERE ACCIDENT DETECTED! Video: {video_name}. Immediate action required."

    payload = {
        "route": "q",
        "message": message,
        "language": "english",
        "numbers": ALERT_PHONE_NUMBER
    }

    headers = {
        "authorization": FAST2SMS_API_KEY,
        "Content-Type": "application/json"
    }

    try:
        response = requests.post(url, json=payload, headers=headers, timeout=10)
        print("[ALERT] SMS sent:", response.json())
    except Exception as e:
        print("[ALERT ERROR] Failed to send SMS:", str(e))


# ============================
# MAIN EXECUTION
# ============================

if len(sys.argv) < 2:
    print("Usage: python -m backend.debug_severity <video_path>")
    sys.exit(1)

video_path = sys.argv[1]

print(f"[INFO] Processing video: {video_path}")

severity, probs_text = predict_severity_ml(video_path)

# ML already prints the probabilities internally, so just print what we got
print(f"Final decision: {severity}")

# ============================
# SEND ALERT IF SEVERE
# ============================

if severity.upper() == "SEVERE":
    print("[ALERT] Sending SMS...")
    send_severe_alert(video_path)
else:
    print("[ALERT] No SMS sent.")
