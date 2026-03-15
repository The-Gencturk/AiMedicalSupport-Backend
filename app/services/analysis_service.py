import os
import cv2
import numpy as np
from sqlalchemy.orm import Session
from fastapi import HTTPException
from app.models.AnalisyModel import Analysis, AnalysisReview, AnalysisStatus
from app.schemas.analisy import ReviewCreate
from app.services.radiology_service import RadiologyService

UPLOAD_DIR = r"C:\Users\LENOVO\Desktop\Projeler\AiMedicalSupport-Backend\uploads"
HEATMAP_DIR = r"C:\Users\LENOVO\Desktop\Projeler\AiMedicalSupport-Backend\uploads\heatmaps"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(HEATMAP_DIR, exist_ok=True)

radiology = RadiologyService()


def create_analysis(db: Session, patient_id: int, doctor_id: int, image_bytes: bytes, filename: str) -> Analysis:
    # Görüntüyü kaydet
    image_path = os.path.join(UPLOAD_DIR, filename)
    with open(image_path, "wb") as f:
        f.write(image_bytes)

    # AI analiz
    result = radiology.analyze(image_bytes)

    # Isı haritası oluştur ve kaydet
    heatmap_path = None
    try:
        np_arr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        img_resized = cv2.resize(img, (224, 224))
        img_norm = img_resized / 255.0
        img_input = np.reshape(img_norm, (1, 224, 224, 3))

        heatmap = radiology._get_heatmap(img_input)
        if heatmap is not None:
            heatmap_filename = f"heatmap_{filename}"
            heatmap_path = os.path.join(HEATMAP_DIR, heatmap_filename)
            heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
            heatmap_uint8 = np.uint8(255 * heatmap_resized)
            heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
            overlay = cv2.addWeighted(heatmap_colored, 0.4, img, 0.6, 0)
            cv2.imwrite(heatmap_path, overlay)
    except:
        pass

    analysis = Analysis(
        patient_id=patient_id,
        doctor_id=doctor_id,
        image_path=image_path,
        heatmap_path=heatmap_path,
        result=result["result"],
        confidence=result["confidence"],
        is_bleeding=result["is_bleeding"],
        status=AnalysisStatus.pending
    )
    db.add(analysis)
    db.commit()
    db.refresh(analysis)
    return analysis


def get_all_analyses(db: Session) -> list[Analysis]:
    return db.query(Analysis).order_by(Analysis.created_at.desc()).all()


def get_analysis(db: Session, analysis_id: int) -> Analysis:
    analysis = db.query(Analysis).filter(Analysis.id == analysis_id).first()
    if not analysis:
        raise HTTPException(status_code=404, detail="Analiz bulunamadı")
    return analysis


def add_review(db: Session, analysis_id: int, doctor_id: int, data: ReviewCreate) -> AnalysisReview:
    analysis = get_analysis(db, analysis_id)
    review = AnalysisReview(
        analysis_id=analysis_id,
        doctor_id=doctor_id,
        label=data.label,
        severity=data.severity,
        note=data.note
    )
    db.add(review)
    analysis.status = AnalysisStatus.reviewed
    db.commit()
    db.refresh(review)
    return review