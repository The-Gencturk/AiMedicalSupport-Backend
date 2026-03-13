from fastapi import APIRouter, UploadFile, File, HTTPException
from app.services.radiology_service import analyze
from app.schemas.analysis_schema import AnalysisResult
from typing import Optional

router = APIRouter()

@router.post("/analyze", response_model=AnalysisResult, summary="Analyze brain radiology image")
async def analyze_image(
    file: UploadFile = File(..., description="Medical image file to analyze")
):
    """
    Analyze brain radiology image using AI model.
    
    - **file**: Medical image file (DICOM, PNG, or JPEG)
    
    Returns analysis result with status and details.
    """
    try:
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="File is empty")
        
        result = analyze(content)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/analyze/info", summary="Get analyze endpoint info")
async def analyze_info():
    """Get information about the analyze endpoint."""
    return {
        "endpoint": "/api/v1/analyze",
        "method": "POST",
        "description": "Analyze brain radiology image",
        "accepts": "multipart/form-data",
        "returns": "AnalysisResult"
    }