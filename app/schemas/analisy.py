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


class BleedingType(str, Enum):
    epidural = "epidural"
    subdural = "subdural"
    subarachnoid = "subaraknoid"
    intraparenchymal = "intraparenkimal"
    intraventricular = "intraventrikuler"
    other = "diger"


class AllAnalysisResponse(BaseModel):
    id: int
    result: str
    confidence: float
    is_bleeding: bool
    bleeding_type: Optional[BleedingType] = None
    status: str

    patient_name: str
    doctor_name: str

    class Config:
        from_attributes = True

class AnalysisResponse(BaseModel):
    id: int
    patient_id: int
    doctor_id: Optional[int]
    image_path: str
    heatmap_path: Optional[str]
    result: str
    confidence: float
    is_bleeding: bool
    bleeding_type: Optional[BleedingType] = None
    status: str
    created_at: datetime

    class Config:
        from_attributes = True

class ReviewCreate(BaseModel):
    is_bleeding: Optional[bool] = None
    bleeding_type: Optional[BleedingType] = None
    label: Optional[str] = None
    severity: Optional[Severity] = Severity.none
    note: Optional[str] = None

    @model_validator(mode="after")
    def validate_bleeding_type(self):
        if self.is_bleeding is None and self.label:
            label_text = self.label.lower()
            if "kanama" in label_text and "yok" not in label_text:
                self.is_bleeding = True
            elif "normal" in label_text or "kanama yok" in label_text:
                self.is_bleeding = False

        if self.is_bleeding is None:
            raise ValueError("is_bleeding zorunludur.")

        if self.is_bleeding and self.bleeding_type is None:
            raise ValueError("Kanama secildiginde bleeding_type zorunludur.")
        if not self.is_bleeding:
            self.bleeding_type = None
        return self

class ReviewResponse(BaseModel):
    id: int
    analysis_id: int
    doctor_id: int
    label: str
    is_bleeding: bool
    bleeding_type: Optional[BleedingType] = None
    model_trained: bool
    severity: Optional[Severity]
    note: Optional[str]
    created_at: datetime

    class Config:
        from_attributes = True

