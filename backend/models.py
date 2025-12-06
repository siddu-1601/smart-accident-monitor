# backend/models.py

from datetime import datetime
from sqlalchemy import (
    Column,
    Integer,
    String,
    Boolean,
    DateTime,
    ForeignKey,
    Text,
    Float,
)
from sqlalchemy.orm import relationship

from .db import Base


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)

    # Adjust these if your existing schema is different
    email = Column(String, unique=True, index=True, nullable=False)
    password_hash = Column(String, nullable=False)
    role = Column(String, default="user", nullable=False)

    incidents = relationship("Incident", back_populates="user")
    cameras = relationship("Camera", back_populates="owner")


class Camera(Base):
    __tablename__ = "cameras"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    rtsp_url = Column(Text, nullable=False)

    location_lat = Column(Float, nullable=True)
    location_lng = Column(Float, nullable=True)

    owner_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    owner = relationship("User", back_populates="cameras")


class Incident(Base):
    __tablename__ = "incidents"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)

    video_path = Column(Text, nullable=False)

    location_lat = Column(Float, nullable=False)
    location_lng = Column(Float, nullable=False)

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # NEW FIELDS FOR ML + ALERTING
    # ----------------------------
    # PENDING / MINOR / MAJOR / SEVERE
    severity = Column(String, default="PENDING", nullable=False)

    # e.g. "rear-end", "head-on", etc. Placeholder for now.
    accident_type = Column(String, nullable=True)

    # Whether video has been processed by the background ML worker
    processed = Column(Boolean, default=False, nullable=False)
    processed_at = Column(DateTime, nullable=True)

    user = relationship("User", back_populates="incidents")
    alerts = relationship("Alert", back_populates="incident", cascade="all, delete-orphan")


class Alert(Base):
    __tablename__ = "alerts"

    id = Column(Integer, primary_key=True, index=True)

    incident_id = Column(Integer, ForeignKey("incidents.id"), nullable=False)
    severity = Column(String, nullable=False)

    # Short human-readable description of the alert
    message = Column(Text, nullable=False)

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    incident = relationship("Incident", back_populates="alerts")
