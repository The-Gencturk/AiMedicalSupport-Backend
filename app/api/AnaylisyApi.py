from fastapi import APIRouter, UploadFile, File, HTTPException
from app.services.radiology_service import RadiologyService
from app.schemas.analysis_schema import AnalysisResult
from typing import Optional


router = APIRouter()
service = RadiologyService()

@router.post("/analyze")
async def analyze_image(
    file: UploadFile = File(...),
    label: Optional[int] = None  # Gönderilirse eğitim de yapar
):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Sadece resim dosyası yükleyebilirsiniz")
    
    image_bytes = await file.read()
    result = service.analyze(image_bytes)
    
    if label is not None:
        if label not in [0, 1]:
            raise HTTPException(status_code=400, detail="Label 0 (NORMAL) veya 1 (KANAMA) olmalı")
        service.train(image_bytes, label)
        result["trained"] = True
        result["trained_label"] = "KANAMA" if label == 1 else "NORMAL"
    
    return result

