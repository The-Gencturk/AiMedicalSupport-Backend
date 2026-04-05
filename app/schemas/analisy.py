from pydantic import BaseModel
from pydantic import model_validator
from typing import Optional
from datetime import datetime
from enum import Enum

class Severity(str, Enum):
    none = "none"
    mild = "hafif"
    moderate = "orta"
    severe = "ciddi"


class AllAnalysisResponse(BaseModel):
    id: int
    scan_type: str
    result: str
    confidence: float
    has_finding: bool
    finding_type: Optional[str] = None
    status: str
    patient_name: str
    doctor_name: str

    class Config:
        from_attributes = True


class AnalysisResponse(BaseModel):
    id: int
    patient_id: Optional[int]
    doctor_id: Optional[int]
    scan_type: str
    image_path: str
    heatmap_path: Optional[str]
    result: str
    confidence: float
    has_finding: bool
    finding_type: Optional[str] = None
    status: str
    created_at: datetime

    class Config:
        from_attributes = True


class ReviewCreate(BaseModel):
    has_finding: Optional[bool] = None
    finding_type: Optional[str] = None
    label: Optional[str] = None
    severity: Optional[Severity] = Severity.none
    note: Optional[str] = None

    @model_validator(mode="before")
    @classmethod
    def validate_fields(cls, data: dict):
        has_finding = data.get("has_finding")
        finding_type = data.get("finding_type")
        label = data.get("label")

        if has_finding is None and label:
            label_text = label.lower()
            if "normal" in label_text:
                has_finding = False
            else:
                has_finding = True
            data["has_finding"] = has_finding

        if has_finding is None:
            raise ValueError("has_finding zorunludur.")

        if not has_finding:
            data["finding_type"] = None
            data["severity"] = "none"

        return data


class ReviewResponse(BaseModel):
    id: int
    analysis_id: int
    doctor_id: int
    label: str
    has_finding: bool
    finding_type: Optional[str] = None
    model_trained: bool
    severity: Optional[Severity]
    note: Optional[str]
    created_at: datetime

    class Config:
        from_attributes = True