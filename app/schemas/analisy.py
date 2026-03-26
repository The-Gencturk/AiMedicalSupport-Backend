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
    diger = "diger"


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

    @model_validator(mode="before")  
    @classmethod
    def validate_bleeding_type(cls, data: dict):
        # 1. Ham veriden değerleri alalım
        is_bleeding = data.get("is_bleeding")
        b_type = data.get("bleeding_type")
        label = data.get("label")

        # 2. Eğer frontend "none" stringi gönderdiyse onu Python None (null) yap
        if b_type == "none":
            data["bleeding_type"] = None
            b_type = None

        # 3. Senin yazdığın label mantığı
        if is_bleeding is None and label:
            label_text = label.lower()
            if "kanama" in label_text and "yok" not in label_text:
                is_bleeding = True
            elif "normal" in label_text or "kanama yok" in label_text:
                is_bleeding = False
            data["is_bleeding"] = is_bleeding

        # 4. Zorunluluk kontrolleri
        if is_bleeding is None:
            raise ValueError("is_bleeding zorunludur.")

        if is_bleeding and b_type is None:
            raise ValueError("Kanama seçildiğinde bleeding_type zorunludur.")
        
        if not is_bleeding:
            data["bleeding_type"] = None
            data["severity"] = "none" # Severity Enum'unda 'none' var, sorun olmaz

        return data

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

