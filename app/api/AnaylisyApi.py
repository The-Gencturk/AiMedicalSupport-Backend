import os
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from app.core.rbac import require_permission
from app.core.Security import get_current_user
from app.db.DbContext import get_db
from typing import Optional
from app.models.User import User
from app.schemas.analisy import ReviewCreate, AllAnalysisResponse
from app.models.AnalisyModel import Analysis
from app.services.analysis_service import (
    create_analysis,
    get_analysis,
    add_review,
    _resolve_upload_path,
    delete_analysis,
)

router = APIRouter()


@router.post("/analyzeCreate", dependencies=[Depends(require_permission("analyze:create"))])
async def analyze_image(
    patient_id: Optional[int] = None,
    file: UploadFile = File(...),
    scan_type: str = "beyin",  
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400,
        detail="Sadece resim dosyası yükleyebilirsiniz")
    image_bytes = await file.read()
    return create_analysis(db, patient_id, current_user.id, image_bytes, file.filename, scan_type)


@router.get("/analyses", response_model=list[AllAnalysisResponse])
def get_analyses(db: Session = Depends(get_db)):

    analyses = db.query(Analysis).all()

    result = []

    for a in analyses:
        result.append({
            "id": a.id,
            "result": a.result,
            "confidence": a.confidence,
            "is_bleeding": a.is_bleeding,
            "bleeding_type": a.bleeding_type,
            "status": a.status,
            
            "patient_name": a.patient.full_name if a.patient else "Bilinmiyor",
            "doctor_name": a.doctor.full_name if a.doctor else "Bilinmiyor"
        })

    return result


@router.get("/analyses/{analysis_id}", dependencies=[Depends(require_permission("analyze:read"))])
def analysis_detail(analysis_id: int, db: Session = Depends(get_db)):
    return get_analysis(db, analysis_id)


@router.post("/analyses/{analysis_id}/review", dependencies=[Depends(require_permission("analyze:read"))])
def review_analysis(
    analysis_id: int,
    data: ReviewCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    return add_review(db, analysis_id, current_user.id, data)


@router.get("/analyses/{analysis_id}/heatmap", dependencies=[Depends(require_permission("analyze:read"))])
def get_heatmap(analysis_id: int, db: Session = Depends(get_db)):
    analysis = get_analysis(db, analysis_id)
    if not analysis.heatmap_path:
        raise HTTPException(status_code=404, detail="Isı haritası bulunamadı")
    full_path = _resolve_upload_path(analysis.heatmap_path)
    if not os.path.exists(full_path):
        raise HTTPException(status_code=404, detail="Heatmap dosyasi bulunamadi")
    return FileResponse(full_path, media_type="image/jpeg")


    
@router.delete("/analyses/{analysis_id}", dependencies=[Depends(require_permission("analyze:delete"))])
def delete_analysis_endpoint(analysis_id: int, db: Session = Depends(get_db)):
    return delete_analysis(db, analysis_id)
