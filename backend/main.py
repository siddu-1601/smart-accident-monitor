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
    Form,
    Depends,
)
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette import status
from sqlalchemy.orm import Session

from .db import get_db, SessionLocal
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
# Background ML classifier (stub) + processing pipeline
# ------------------------------------------------------------------------------

def classify_severity_stub(video_path: str) -> tuple[str, str]:
    """
    Placeholder ML classifier.

    Returns (severity, accident_type).
    Replace this later with a real ML model.
    """
    # Current policy: treat all as MINOR to avoid spam.
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
# HOME / OVERVIEW
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
# USER REGISTRATION
# ------------------------------------------------------------------------------

@app.get("/register", response_class=HTMLResponse)
async def get_register(request: Request):
    return templates.TemplateResponse(
        "register.html",
        {"request": request},
    )


@app.post("/register")
async def post_register(
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
# CCTV CAMERA REGISTRATION
# ------------------------------------------------------------------------------

@app.get("/add-camera", response_class=HTMLResponse)
async def get_add_camera(request: Request, db: Session = Depends(get_db)):
    # Optional: pass users to template for a dropdown
    users: List[User] = db.query(User).all()

    return templates.TemplateResponse(
        "add_camera.html",
        {
            "request": request,
            "users": users,
        },
    )


@app.post("/add-camera")
async def post_add_camera(
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
# UPLOAD INCIDENT PAGE (Uber-style UI)
# ------------------------------------------------------------------------------

@app.get("/upload-incident", response_class=HTMLResponse)
async def get_upload_incident(request: Request):
    """
    Renders upload_incident.html which already has:
    - Record Now / Choose from Gallery
    - Leaflet map
    - 'Use my current location'
    DO NOT touch the frontend logic; only keep backend endpoints stable.
    """
    return templates.TemplateResponse(
        "upload_incident.html",
        {"request": request},
    )


# ------------------------------------------------------------------------------
# SHARED INCIDENT CREATION HANDLER
# ------------------------------------------------------------------------------

async def _handle_incident_create(
    request: Request,
    background_tasks: BackgroundTasks,
    db: Session,
):
    """
    Central handler that:
    - reads multipart form data
    - saves video under /static/uploads
    - inserts Incident row with severity=PENDING
    - kicks background worker to classify and update severity
    This is wired to multiple POST routes so frontend URL mismatches
    don't cause 404 "detail: Not Found".
    """

    form = await request.form()

    # File field: try common names
    file_obj = (
        form.get("file")
        or form.get("video")
        or form.get("upload")
    )
    if not isinstance(file_obj, UploadFile):
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"detail": "Video file is required"},
        )

    # Location fields
    raw_lat = form.get("lat") or form.get("location_lat")
    raw_lng = form.get("lng") or form.get("location_lng")

    if raw_lat is None or raw_lng is None:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"detail": "Latitude and longitude are required"},
        )

    try:
        lat = float(raw_lat)
        lng = float(raw_lng)
    except ValueError:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"detail": "Invalid latitude/longitude"},
        )

    # user_id (required because DB schema has NOT NULL)
    raw_user_id = form.get("user_id")
    if raw_user_id is None or raw_user_id == "":
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"detail": "user_id is required"},
        )

    try:
        user_id = int(raw_user_id)
    except ValueError:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"detail": "user_id must be an integer"},
        )

    # Persist file to disk (under /static/uploads so it can be served)
    filename = f"{datetime.utcnow().strftime('%Y%m%d%H%M%S')}_{file_obj.filename}"
    file_path = os.path.join(UPLOAD_DIR, filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file_obj.file, buffer)

    relative_video_path = f"/static/uploads/{filename}"

    # Create Incident row with PENDING severity; worker will flip to MINOR
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

    # Trigger background processing
    background_tasks.add_task(process_incident_video, incident.id)

    return JSONResponse(
        status_code=status.HTTP_201_CREATED,
        content={
            "status": "ok",
            "incident_id": incident.id,
            "video_path": incident.video_path,
        },
    )


# ------------------------------------------------------------------------------
# INCIDENT CREATION ENDPOINTS (multiple paths to avoid 404)
# ------------------------------------------------------------------------------

@app.post("/incidents")
async def create_incident(
    request: Request,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
):
    return await _handle_incident_create(request, background_tasks, db)


@app.post("/api/incidents")
async def create_incident_api(
    request: Request,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
):
    # In case frontend is calling /api/incidents
    return await _handle_incident_create(request, background_tasks, db)


@app.post("/upload-incident")
async def create_incident_legacy(
    request: Request,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
):
    # In case the form or JS posts back to /upload-incident
    return await _handle_incident_create(request, background_tasks, db)


# ------------------------------------------------------------------------------
# ALERTS LISTING (JSON API)
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
