from pydantic import BaseModel
from typing import Optional
from datetime import datetime
from enum import Enum

class Severity(str, Enum):
    none = "none"
    mild = "hafif"
    moderate = "orta"
    severe = "ciddi"

class AnalysisResponse(BaseModel):
    id: int
    patient_id: int
    doctor_id: Optional[int]
    image_path: str
    heatmap_path: Optional[str]
    result: str
    confidence: float
    is_bleeding: bool
    status: str
    created_at: datetime

    class Config:
        from_attributes = True

class ReviewCreate(BaseModel):
    label: str
    severity: Optional[Severity] = Severity.none
    note: Optional[str] = None

class ReviewResponse(BaseModel):
    id: int
    analysis_id: int
    doctor_id: int
    label: str
    severity: Optional[Severity]
    note: Optional[str]
    created_at: datetime

    class Config:
        from_attributes = True