# backend/main.py

import os
from datetime import datetime

from fastapi import FastAPI, Request, Form, UploadFile, File
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session

from .db import Base, engine, get_db
from .models import User, UserRole, Camera, Incident, IncidentSeverity

# Create all tables in the database (runs once at startup)
Base.metadata.create_all(bind=engine)

app = FastAPI(title="Smart Accident Monitor")

# Mount static directory (not heavily used yet)
static_dir = os.path.join("backend", "static")
if not os.path.exists(static_dir):
    os.makedirs(static_dir, exist_ok=True)

app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Templates directory
templates = Jinja2Templates(directory="backend/templates")


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# ---------------- USER REGISTRATION ----------------

@app.get("/register", response_class=HTMLResponse)
def show_register(request: Request):
    return templates.TemplateResponse("register.html", {"request": request, "error": None})


@app.post("/register", response_class=HTMLResponse)
def register_user(
    request: Request,
    name: str = Form(...),
    email: str = Form(...),
    password: str = Form(...),
    role: str = Form(...)
):
    db: Session = next(get_db())

    existing = db.query(User).filter(User.email == email).first()
    if existing:
        return templates.TemplateResponse(
            "register.html",
            {
                "request": request,
                "error": "Email already registered. Use a different email."
            }
        )

    user_role = UserRole(role)

    user = User(
        name=name,
        email=email,
        password=password,  # plain text (OK for academic project, not for real life)
        role=user_role
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    return RedirectResponse(url="/", status_code=302)


# ---------------- USER INCIDENT UPLOAD ----------------

UPLOAD_DIR = "uploaded_videos"
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR, exist_ok=True)


@app.get("/upload-incident", response_class=HTMLResponse)
def show_upload_incident(request: Request):
    return templates.TemplateResponse(
        "upload_incident.html",
        {"request": request, "message": None}
    )


@app.post("/upload-incident", response_class=HTMLResponse)
async def upload_incident(
    request: Request,
    video: UploadFile = File(...),
    location_lat: float = Form(...),
    location_lng: float = Form(...),
    description: str = Form(default="")
):
    # Save the uploaded video file
    timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    filename = f"{timestamp}_{video.filename}"
    filepath = os.path.join(UPLOAD_DIR, filename)

    with open(filepath, "wb") as f:
        content = await video.read()
        f.write(content)

    # Create Incident in DB
    db: Session = next(get_db())

    incident = Incident(
        source="USER_UPLOAD",
        camera_id=None,
        user_id=None,  # later we will link to logged-in user
        location_lat=location_lat,
        location_lng=location_lng,
        occurred_at=datetime.utcnow(),
        severity=IncidentSeverity.PENDING,
        accident_type=None,
        video_path=filepath,
    )
    db.add(incident)
    db.commit()
    db.refresh(incident)

    message = f"Incident received with ID {incident.id}. Processing will happen in background."
    return templates.TemplateResponse(
        "upload_incident.html",
        {"request": request, "message": message}
    )


# ---------------- CCTV CAMERA REGISTRATION ----------------

@app.get("/add-camera", response_class=HTMLResponse)
def show_add_camera(request: Request):
    return templates.TemplateResponse(
        "add_camera.html",
        {"request": request, "message": None, "error": None}
    )


@app.post("/add-camera", response_class=HTMLResponse)
def add_camera(
    request: Request,
    owner_id: int = Form(...),
    name: str = Form(...),
    rtsp_url: str = Form(...),
    location_lat: float = Form(...),
    location_lng: float = Form(...)
):
    db: Session = next(get_db())

    owner = db.query(User).filter(User.id == owner_id).first()
    if not owner:
        return templates.TemplateResponse(
            "add_camera.html",
            {"request": request, "message": None, "error": "Owner user ID not found."}
        )

    camera = Camera(
        owner_id=owner_id,
        name=name,
        rtsp_url=rtsp_url,
        location_lat=location_lat,
        location_lng=location_lng
    )
    db.add(camera)
    db.commit()
    db.refresh(camera)

    msg = f"Camera registered with ID {camera.id}"
    return templates.TemplateResponse(
        "add_camera.html",
        {"request": request, "message": msg, "error": None}
    )
