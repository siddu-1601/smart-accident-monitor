# backend/main.py

import os
import shutil
from datetime import datetime
from typing import Optional, List

from fastapi import (
    FastAPI,
    Request,
    Form,
    UploadFile,
    File,
    Depends,
    BackgroundTasks,
)
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette import status
from sqlalchemy.orm import Session

from .db import get_db, SessionLocal
from .models import User, Camera, Incident, Alert


# ------------------------------------------------------------------------------
# Core app / static / templates
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
# Helper: ensure an anonymous user exists
# ------------------------------------------------------------------------------

def get_or_create_anonymous_user(db: Session) -> User:
    """
    Ensures there is a dummy 'anonymous' user so incidents can be saved
    even when the front-end does not send user_id.
    This avoids NOT NULL errors on incidents.user_id.
    """
    email = "anonymous@smart-accident-monitor.local"
    user = db.query(User).filter(User.email == email).first()
    if user:
        return user

    # Password is meaningless here; it's just to satisfy NOT NULL
    user = User(
        email=email,
        password_hash="!",
        role="anonymous",
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


# ------------------------------------------------------------------------------
# Stub ML classifier + background worker
# ------------------------------------------------------------------------------

def classify_severity_stub(video_path: str) -> tuple[str, str]:
    """
    Stub ML classifier.
    For now, always returns MINOR so no alerts are generated.
    """
    severity = "MINOR"
    accident_type = "generic"
    return severity, accident_type


def process_incident_video(incident_id: int) -> None:
    """
    Background task:
    - Load Incident
    - Run stub classifier
    - Update severity/processed fields
    - Create Alert only for MAJOR/SEVERE (will not happen with stub)
    """
    db: Session = SessionLocal()
    try:
        incident: Optional[Incident] = (
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
# Home / overview
# ------------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
def read_root(request: Request, db: Session = Depends(get_db)):
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
# User registration
# ------------------------------------------------------------------------------

@app.get("/register", response_class=HTMLResponse)
def get_register(request: Request):
    return templates.TemplateResponse(
        "register.html",
        {"request": request},
    )


@app.post("/register")
def post_register(
    request: Request,
    email: str = Form(...),
    password: str = Form(...),
    role: str = Form("user"),
    db: Session = Depends(get_db),
):
    import hashlib

    password_hash = hashlib.sha256(password.encode()).hexdigest()

    user = User(email=email, password_hash=password_hash, role=role)
    db.add(user)
    db.commit()
    db.refresh(user)

    return RedirectResponse(
        url="/",
        status_code=status.HTTP_303_SEE_OTHER,
    )


# ------------------------------------------------------------------------------
# CCTV camera registration
# ------------------------------------------------------------------------------

@app.get("/add-camera", response_class=HTMLResponse)
def get_add_camera(request: Request, db: Session = Depends(get_db)):
    users: List[User] = db.query(User).all()

    return templates.TemplateResponse(
        "add_camera.html",
        {
            "request": request,
            "users": users,
        },
    )


@app.post("/add-camera")
def post_add_camera(
    request: Request,
    name: str = Form(...),
    rtsp_url: str = Form(...),
    location_lat: float = Form(...),
    location_lng: float = Form(...),
    owner_id: int = Form(...),
    db: Session = Depends(get_db),
):
    camera = Camera(
        name=name,
        rtsp_url=rtsp_url,
        location_lat=location_lat,
        location_lng=location_lng,
        owner_id=owner_id,
    )
    db.add(camera)
    db.commit()
    db.refresh(camera)

    return RedirectResponse(
        url="/",
        status_code=status.HTTP_303_SEE_OTHER,
    )


# ------------------------------------------------------------------------------
# Upload incident page (Uber-style UI) – GET
# ------------------------------------------------------------------------------

@app.get("/upload-incident", response_class=HTMLResponse)
def get_upload_incident(request: Request):
    """
    Renders upload_incident.html which you already shared.
    Form posts back to the same URL (no action attribute).
    """
    message = request.query_params.get("message")
    error = request.query_params.get("error")

    return templates.TemplateResponse(
        "upload_incident.html",
        {
            "request": request,
            "message": message,
            "error": error,
        },
    )


# ------------------------------------------------------------------------------
# Upload incident – POST (matches your HTML exactly)
# ------------------------------------------------------------------------------

@app.post("/upload-incident", response_class=HTMLResponse)
async def post_upload_incident(
    request: Request,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),

    # This matches your JS which sets name="video" on the active input
    video: UploadFile = File(...),

    # Hidden fields in your form
    location_lat: float = Form(...),
    location_lng: float = Form(...),

    # Optional description textarea
    description: Optional[str] = Form(None),
):
    """
    Handles the Uber-style upload flow:

    - Accepts the 'video' file field (camera/gallery toggle already sets name="video").
    - Reads location_lat and location_lng from hidden inputs.
    - Uses or creates an 'anonymous' user so user_id is never NULL.
    - Inserts Incident with severity=PENDING; background worker sets MINOR.
    """

    # 1) Ensure we have an anonymous user
    anon_user = get_or_create_anonymous_user(db)

    # 2) Save the video file under /static/uploads
    filename = f"{datetime.utcnow().strftime('%Y%m%d%H%M%S')}_{video.filename}"
    file_path = os.path.join(UPLOAD_DIR, filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(video.file, buffer)

    # Path that can be served via /static
    relative_video_path = f"/static/uploads/{filename}"

    # 3) Insert incident row
    incident = Incident(
        user_id=anon_user.id,
        video_path=relative_video_path,
        location_lat=location_lat,
        location_lng=location_lng,
        severity="PENDING",
        accident_type=None,
        processed=False,
        processed_at=None,
    )
    db.add(incident)
    db.commit()
    db.refresh(incident)

    # 4) Kick background processing
    background_tasks.add_task(process_incident_video, incident.id)

    # 5) Redirect back with success message
    success_msg = "Incident submitted successfully. It will be processed shortly."
    url = f"/upload-incident?message={success_msg}"

    return RedirectResponse(
        url=url,
        status_code=status.HTTP_303_SEE_OTHER,
    )


# ------------------------------------------------------------------------------
# Alerts listing
# ------------------------------------------------------------------------------

@app.get("/alerts", response_class=HTMLResponse)
def get_alerts(request: Request, db: Session = Depends(get_db)):
    alerts: List[Alert] = (
        db.query(Alert)
        .order_by(Alert.created_at.desc())
        .limit(50)
        .all()
    )

    return templates.TemplateResponse(
        "alerts.html",
        {
            "request": request,
            "alerts": alerts,
        },
    )
