# backend/main.py

import os
import shutil
from datetime import datetime
from typing import List

from fastapi import (
    FastAPI,
    Request,
    BackgroundTasks,
    UploadFile,
    File,
    Form,
    Depends,
)
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from sqlalchemy.orm import Session

from .db import SessionLocal
from .models import User, Camera, Incident, Alert


# ------------------------------------------------------------------------------
# FastAPI app + Jinja + static
# ------------------------------------------------------------------------------

app = FastAPI(title="Smart Accident Monitor")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
STATIC_DIR = os.path.join(BASE_DIR, "static")
UPLOAD_DIR = os.path.join(STATIC_DIR, "uploads")

os.makedirs(UPLOAD_DIR, exist_ok=True)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=TEMPLATES_DIR)


# ------------------------------------------------------------------------------
# DB session dependency
# ------------------------------------------------------------------------------

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ------------------------------------------------------------------------------
# Background ML classifier (stub) + processing pipeline
# ------------------------------------------------------------------------------

def classify_severity_stub(video_path: str) -> tuple[str, str]:
    """
    Placeholder ML classifier.
    Returns (severity, accident_type).

    Replace this later with a real ML model loading from disk / cloud.
    """
    # TODO: Implement real ML. For now, simple deterministic behavior.
    # Example policy:
    # - Treat all as MINOR to avoid spam
    severity = "MINOR"
    accident_type = "generic"
    return severity, accident_type


def process_incident_video(incident_id: int) -> None:
    """
    Background task:
    - Fetch incident
    - Run classifier (stub for now)
    - Update DB with severity + accident_type
    - Create alerts for MAJOR / SEVERE
    """
    db: Session = SessionLocal()
    try:
        incident: Incident | None = (
            db.query(Incident).filter(Incident.id == incident_id).first()
        )
        if not incident:
            return

        severity, accident_type = classify_severity_stub(incident.video_path)

        incident.severity = severity
        incident.accident_type = accident_type
        incident.processed = True
        incident.processed_at = datetime.utcnow()

        if severity in ("MAJOR", "SEVERE"):
            msg = (
                f"Accident ({severity}) detected at "
                f"({incident.location_lat}, {incident.location_lng})"
            )
            alert = Alert(
                incident_id=incident.id,
                severity=severity,
                message=msg,
            )
            db.add(alert)

        db.commit()
    finally:
        db.close()


# ------------------------------------------------------------------------------
# Simple homepage (you can beautify later)
# ------------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request, db: Session = Depends(get_db)):
    incidents: List[Incident] = (
        db.query(Incident)
        .order_by(Incident.created_at.desc())
        .limit(10)
        .all()
    )
    alerts: List[Alert] = (
        db.query(Alert)
        .order_by(Alert.created_at.desc())
        .limit(10)
        .all()
    )

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "incidents": incidents,
            "alerts": alerts,
        },
    )


# ------------------------------------------------------------------------------
# Upload incident page (your Uber-style map UI already lives in this template)
# ------------------------------------------------------------------------------

@app.get("/upload-incident", response_class=HTMLResponse)
async def get_upload_incident(request: Request):
    """
    Renders upload_incident.html which already has:
    - Record Now / Choose from Gallery
    - Leaflet map
    - Use my current location
    - User must pick location before submit
    """
    return templates.TemplateResponse("upload_incident.html", {"request": request})


# ------------------------------------------------------------------------------
# Incident creation endpoint (called from upload_incident.html form)
# ------------------------------------------------------------------------------

@app.post("/incidents")
async def create_incident(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    lat: float = Form(...),
    lng: float = Form(...),
    user_id: int = Form(...),  # adjust if you use session auth
    db: Session = Depends(get_db),
):
    """
    Creates an incident from a video upload and location.
    - Saves video under static/uploads
    - Inserts Incident row with severity=PENDING
    - Triggers background processing to classify severity & create alerts
    """

    # 1) Persist file to disk
    filename = f"{datetime.utcnow().strftime('%Y%m%d%H%M%S')}_{file.filename}"
    file_path = os.path.join(UPLOAD_DIR, filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Convert to relative path under /static so frontend can access it
    relative_video_path = f"/static/uploads/{filename}"

    # 2) Create incident row
    incident = Incident(
        user_id=user_id,
        video_path=relative_video_path,
        location_lat=lat,
        location_lng=lng,
        severity="PENDING",
        processed=False,
    )

    db.add(incident)
    db.commit()
    db.refresh(incident)

    # 3) Trigger background processing
    background_tasks.add_task(process_incident_video, incident.id)

    return {
        "status": "ok",
        "incident_id": incident.id,
        "video_path": incident.video_path,
    }


# ------------------------------------------------------------------------------
# Alerts listing (basic API for now, you can convert to HTML dashboard later)
# ------------------------------------------------------------------------------

@app.get("/alerts")
async def list_alerts(db: Session = Depends(get_db)):
    alerts: List[Alert] = (
        db.query(Alert)
        .order_by(Alert.created_at.desc())
        .limit(50)
        .all()
    )
    return [
        {
            "id": a.id,
            "incident_id": a.incident_id,
            "severity": a.severity,
            "message": a.message,
            "created_at": a.created_at.isoformat(),
        }
        for a in alerts
    ]
