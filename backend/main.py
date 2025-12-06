# backend/main.py

import os
import shutil
from datetime import datetime
from typing import List, Optional

from fastapi import (
    FastAPI,
    Request,
    BackgroundTasks,
    UploadFile,
    File,
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
    # For now, avoid noise: treat everything as MINOR / generic
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
# Overview / home
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
# Upload incident page (Uber-style map flow)
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
    return templates.TemplateResponse(
        "upload_incident.html",
        {
            "request": request,
            "message": None,
            "error": None,
        },
    )


# ------------------------------------------------------------------------------
# Incident creation endpoint (called from upload_incident.html form)
# ------------------------------------------------------------------------------

@app.post("/incidents")
async def create_incident(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    lat: float = Depends(),
    lng: float = Depends(),
    user_id: int = Depends(),
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
# Alerts listing (basic JSON API)
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


# ------------------------------------------------------------------------------
# User registration (RESTORES /register)
# ------------------------------------------------------------------------------

@app.get("/register", response_class=HTMLResponse)
async def get_register_user(request: Request):
    """
    Render the registration form.
    """
    return templates.TemplateResponse(
        "register.html",
        {
            "request": request,
            "error": None,
            "success": None,
        },
    )


@app.post("/register", response_class=HTMLResponse)
async def post_register_user(request: Request, db: Session = Depends(get_db)):
    """
    Handle user registration form POST.
    We read the form generically to be robust against field-name changes.
    """
    form = await request.form()

    full_name = form.get("full_name") or form.get("name") or ""
    email = (form.get("email") or "").strip()
    password = form.get("password") or form.get("pass") or ""
    role_raw = form.get("role") or form.get("user_type") or "user"

    if not email or not password:
        return templates.TemplateResponse(
            "register.html",
            {
                "request": request,
                "error": "Email and password are required.",
                "success": None,
            },
        )

    # Normalize role value
    role_raw_lower = role_raw.lower()
    if "cctv" in role_raw_lower:
        role = "cctv_owner"
    else:
        role = "user"

    # Check duplicate email
    existing = db.query(User).filter(User.email == email).first()
    if existing:
        return templates.TemplateResponse(
            "register.html",
            {
                "request": request,
                "error": "A user with this email already exists.",
                "success": None,
            },
        )

    user = User(
        email=email,
        password_hash=password,  # plain-text for now; prototype only
        role=role,
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    success_msg = (
        f"User created successfully with ID {user.id} "
        f"({role}). Use this ID to link incidents or CCTV cameras."
    )

    # full_name is currently not stored; we can add a column later if needed.
    return templates.TemplateResponse(
        "register.html",
        {
            "request": request,
            "error": None,
            "success": success_msg,
        },
    )


# ------------------------------------------------------------------------------
# CCTV registration (RESTORES /add-camera)
# ------------------------------------------------------------------------------

@app.get("/add-camera", response_class=HTMLResponse)
async def get_add_camera(request: Request):
    """
    Render CCTV registration form.
    """
    return templates.TemplateResponse(
        "add_camera.html",
        {
            "request": request,
            "error": None,
            "success": None,
        },
    )


@app.post("/add-camera", response_class=HTMLResponse)
async def post_add_camera(request: Request, db: Session = Depends(get_db)):
    """
    Handle CCTV camera registration form POST.
    """
    form = await request.form()

    owner_id_raw = form.get("owner_id") or form.get("owner_user_id") or ""
    name = (form.get("name") or form.get("camera_name") or "").strip()
    rtsp_url = (form.get("rtsp_url") or form.get("rtsp") or "").strip()
    lat_raw = form.get("lat") or form.get("latitude") or ""
    lng_raw = form.get("lng") or form.get("longitude") or ""

    # Basic validation
    try:
        owner_id = int(owner_id_raw)
    except (TypeError, ValueError):
        return templates.TemplateResponse(
            "add_camera.html",
            {
                "request": request,
                "error": "Invalid owner user ID.",
                "success": None,
            },
        )

    if not name or not rtsp_url or not lat_raw or not lng_raw:
        return templates.TemplateResponse(
            "add_camera.html",
            {
                "request": request,
                "error": "All fields are required.",
                "success": None,
            },
        )

    try:
        lat = float(lat_raw)
        lng = float(lng_raw)
    except ValueError:
        return templates.TemplateResponse(
            "add_camera.html",
            {
                "request": request,
                "error": "Latitude and longitude must be numeric.",
                "success": None,
            },
        )

    owner = db.query(User).filter(User.id == owner_id).first()
    if not owner:
        return templates.TemplateResponse(
            "add_camera.html",
            {
                "request": request,
                "error": "Owner user ID not found. Create the CCTV owner user first.",
                "success": None,
            },
        )

    camera = Camera(
        owner_id=owner_id,
        name=name,
        rtsp_url=rtsp_url,
        location_lat=lat,
        location_lng=lng,
    )
    db.add(camera)
    db.commit()
    db.refresh(camera)

    success_msg = (
        f"Camera '{camera.name}' registered with ID {camera.id} "
        f"for owner user {owner_id}."
    )

    return templates.TemplateResponse(
        "add_camera.html",
        {
            "request": request,
            "error": None,
            "success": success_msg,
        },
    )
