from sqlalchemy import Column, Integer, String, Boolean, Float, DateTime, ForeignKey, Text, Enum
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.db.DbContext import Base
import enum

class AnalysisStatus(str, enum.Enum):
    pending = "pending"
    reviewed = "reviewed"
    trained = "trained"

class Severity(str, enum.Enum):
    none = "none"
    mild = "hafif"
    moderate = "orta"
    severe = "ciddi"



class Analysis(Base):
    __tablename__ = "analyses"

    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(Integer, ForeignKey("patients.id", ondelete="CASCADE"), nullable=True)
    doctor_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    scan_type = Column(String, default="brain", nullable=False)
    image_path = Column(String, nullable=False)
    heatmap_path = Column(String, nullable=True)
    result = Column(String, nullable=False)
    confidence = Column(Float, nullable=False)
    has_finding = Column(Boolean, nullable=False)        # is_bleeding → has_finding
    finding_type = Column(String, nullable=True)         # bleeding_type → finding_type
    status = Column(Enum(AnalysisStatus), default=AnalysisStatus.pending)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    patient = relationship("Patient", back_populates="analyses")
    doctor = relationship("User", foreign_keys=[doctor_id])
    reviews = relationship("AnalysisReview", back_populates="analysis", cascade="all, delete-orphan")


class AnalysisReview(Base):
    __tablename__ = "analysis_reviews"

    id = Column(Integer, primary_key=True, index=True)
    analysis_id = Column(Integer, ForeignKey("analyses.id", ondelete="CASCADE"), nullable=False)
    doctor_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    label = Column(String, nullable=False)
    has_finding = Column(Boolean, nullable=False, default=False)   # is_bleeding → has_finding
    finding_type = Column(String, nullable=True)                   # bleeding_type → finding_type
    model_trained = Column(Boolean, nullable=False, default=False)
    severity = Column(Enum(Severity), default=Severity.none)
    note = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    analysis = relationship("Analysis", back_populates="reviews")
    doctor = relationship("User", foreign_keys=[doctor_id])
