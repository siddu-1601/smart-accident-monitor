# backend/models.py

from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, DateTime, Enum, ForeignKey
from sqlalchemy.orm import relationship
import enum

from .db import Base


class UserRole(str, enum.Enum):
    USER = "USER"
    CCTV_OWNER = "CCTV_OWNER"


class IncidentSeverity(str, enum.Enum):
    PENDING = "PENDING"
    MINOR = "MINOR"
    MAJOR = "MAJOR"
    SEVERE = "SEVERE"
    NO_INCIDENT = "NO_INCIDENT"


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    email = Column(String, unique=True, index=True, nullable=False)
    password = Column(String, nullable=False)  # For project only; in real life, hash it
    role = Column(Enum(UserRole), default=UserRole.USER, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    cameras = relationship("Camera", back_populates="owner")
    incidents = relationship("Incident", back_populates="user")


class Camera(Base):
    __tablename__ = "cameras"

    id = Column(Integer, primary_key=True, index=True)
    owner_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    name = Column(String, nullable=False)
    rtsp_url = Column(String, nullable=False)
    location_lat = Column(Float, nullable=False)
    location_lng = Column(Float, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    owner = relationship("User", back_populates="cameras")
    incidents = relationship("Incident", back_populates="camera")


class Incident(Base):
    __tablename__ = "incidents"

    id = Column(Integer, primary_key=True, index=True)

    # "CCTV" or "USER_UPLOAD"
    source = Column(String, nullable=False)

    camera_id = Column(Integer, ForeignKey("cameras.id"), nullable=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)

    location_lat = Column(Float, nullable=False)
    location_lng = Column(Float, nullable=False)
    occurred_at = Column(DateTime, default=datetime.utcnow)

    severity = Column(Enum(IncidentSeverity), default=IncidentSeverity.PENDING)
    accident_type = Column(String, nullable=True)
    video_path = Column(String, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)

    camera = relationship("Camera", back_populates="incidents")
    user = relationship("User", back_populates="incidents")
