from fastapi import APIRouter, UploadFile, File, HTTPException
from app.services.radiology_service import RadiologyService
from app.schemas.analysis_schema import AnalysisResult
from typing import Optional


router = APIRouter()
service = RadiologyService()

@router.post("/analyze")
async def analyze_image(file: UploadFile = File(...)):
    # Sadece resim kabul et
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Sadece resim dosyası yükleyebilirsiniz")
    
    image_bytes = await file.read()
    result = service.analyze(image_bytes)
    return result