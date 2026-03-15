from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from typing import Optional
from app.core.rbac import require_permission
from app.core.Security import get_current_user
from app.db.DbContext import get_db
from app.models.User import User
from app.schemas.analisy import ReviewCreate, AnalysisResponse, ReviewResponse,AllAnalysisResponse
from app.services.analysis_service import (
    create_analysis, get_all_analyses, get_analysis, add_review
)

router = APIRouter()


@router.post("/analyze", dependencies=[Depends(require_permission("analyze:create"))])
async def analyze_image(
    patient_id: int,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Sadece resim dosyası yükleyebilirsiniz")
    image_bytes = await file.read()
    return create_analysis(db, patient_id, current_user.id, image_bytes, file.filename)


@router.get("/analyses", response_model=list[AllAnalysisResponse], dependencies=[Depends(require_permission("analyze:read"))])
def list_analyses(db: Session = Depends(get_db)):
    return get_all_analyses(db)


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
    return FileResponse(analysis.heatmap_path, media_type="image/jpeg")