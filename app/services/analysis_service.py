import os
import cv2
import shutil
import numpy as np
import traceback
from typing import Optional
from sqlalchemy.orm import Session
from fastapi import HTTPException
from app.models.AnalisyModel import Analysis, AnalysisReview, AnalysisStatus
from app.schemas.analisy import ReviewCreate,Severity
from app.services.classification_service import ClassificationService

UPLOAD_DIR = r"C:\Users\LENOVO\Desktop\Projeler\AiMedicalSupport-Backend\uploads"
HEATMAP_DIR = r"C:\Users\LENOVO\Desktop\Projeler\AiMedicalSupport-Backend\uploads\heatmaps"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(HEATMAP_DIR, exist_ok=True)



def _resolve_upload_path(relative_path: str) -> str:
    normalized = relative_path.replace("\\", "/").strip("/")
    if normalized.lower().startswith("uploads/"):
        normalized = normalized[len("uploads/"):]
    candidate = os.path.abspath(os.path.join(UPLOAD_DIR, normalized.replace("/", os.sep)))
    upload_root = os.path.abspath(UPLOAD_DIR)
    if os.path.commonpath([candidate, upload_root]) != upload_root:
        raise HTTPException(status_code=400, detail="Gecersiz dosya yolu.")
    return candidate

def _build_review_label(has_finding: bool, finding_type: Optional[str], label: Optional[str]) -> str:
    if label:
        return label
    if not has_finding:
        return "NORMAL"
    if finding_type:
        return f"ANOMALİ ({finding_type})"
    return "ANOMALİ"


def get_all_analyses(db: Session) -> list[Analysis]:
    return db.query(Analysis).order_by(Analysis.created_at.desc()).all()


def get_analysis(db: Session, analysis_id: int) -> Analysis:
    analysis = db.query(Analysis).filter(Analysis.id == analysis_id).first()
    if not analysis:
        raise HTTPException(status_code=404, detail="Analiz bulunamadı")
    return analysis

def delete_analysis(db: Session, analysis_id: int):
    analysis = get_analysis(db, analysis_id)
    analiz_dir = os.path.join(UPLOAD_DIR, f"analiz_{analysis_id}")
    if os.path.exists(analiz_dir):
        shutil.rmtree(analiz_dir, ignore_errors=True)
    db.delete(analysis)
    db.commit()
    return {"message": "Analiz silindi"}



def create_analysis(db: Session, patient_id: int, doctor_id: int, image_bytes: bytes, filename: str, scan_type: str = "brain") -> Analysis:

    radiology = ClassificationService().get_service(scan_type)
    result = radiology.analyze(image_bytes)
    has_finding = bool(result.get("finding", result.get("is_bleeding", False)))
    finding_type = result.get("finding_type") or result.get("bleeding_type")

    analysis = Analysis(
        patient_id=patient_id,
        doctor_id=doctor_id,
        image_path="pending",
        heatmap_path=None,
        result=result["result"],
        confidence=result["confidence"],
        has_finding=has_finding,
        finding_type=finding_type,
        status=AnalysisStatus.pending,
        scan_type=scan_type
    )
    db.add(analysis)
    db.flush()

    analiz_dir = os.path.join(UPLOAD_DIR, f"analiz_{analysis.id}")
    os.makedirs(analiz_dir, exist_ok=True)

    image_disk_path = os.path.join(analiz_dir, filename)
    with open(image_disk_path, "wb") as f:
        f.write(image_bytes)

    heatmap_url = None
    try:
        np_arr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("cv2.imdecode başarısız")

        img_resized = cv2.resize(img, (224, 224))
        img_norm = img_resized / 255.0
        img_input = np.reshape(img_norm, (1, 224, 224, 3))

        heatmap = radiology._get_heatmap(img_input)
        if heatmap is not None:
            heatmap_filename = f"heatmap_{filename}"
            heatmap_disk_path = os.path.join(analiz_dir, heatmap_filename)
            heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
            heatmap_uint8 = np.uint8(255 * heatmap_resized)
            heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
            overlay = cv2.addWeighted(heatmap_colored, 0.4, img, 0.6, 0)
            cv2.imwrite(heatmap_disk_path, overlay)
            heatmap_url = f"/uploads/analiz_{analysis.id}/{heatmap_filename}"

    except Exception as e:
        print(f"HEATMAP HATA: {e}")
        traceback.print_exc()

    analysis.image_path = f"/uploads/analiz_{analysis.id}/{filename}"
    analysis.heatmap_path = heatmap_url
    db.commit()
    db.refresh(analysis)
    return analysis


def add_review(db: Session, analysis_id: int, doctor_id: int, data: ReviewCreate) -> AnalysisReview:
    analysis = get_analysis(db, analysis_id)
    radiology = ClassificationService().get_service(analysis.scan_type or "brain")

    if not data.has_finding:
        final_finding_type = None
        final_severity = Severity.none
    else:
        final_finding_type = data.finding_type if data.finding_type else None
        final_severity = data.severity
        if not final_finding_type:
            raise HTTPException(status_code=400, detail="Anomali durumunda tür belirtilmelidir.")

    review_label = _build_review_label(
        has_finding=data.has_finding,
        finding_type=final_finding_type,
        label=data.label
    )

    model_trained = False
    image_disk_path = _resolve_upload_path(analysis.image_path)

    if os.path.exists(image_disk_path):
        with open(image_disk_path, "rb") as f:
            image_bytes = f.read()

        train_result = radiology.train(
            image_bytes=image_bytes,
            label=1 if data.has_finding else 0,
            finding_type=final_finding_type,
            bleeding_type=final_finding_type,
            analysis_id=analysis_id,
            doctor_id=doctor_id,
        )
        model_trained = bool(train_result.get("success", False))
        if not model_trained:
            print(f"MODEL TRAIN FAILED id={analysis_id}: {train_result.get('message')}")

    review = AnalysisReview(
        analysis_id=analysis_id,
        doctor_id=doctor_id,
        label=review_label,
        has_finding=data.has_finding,
        finding_type=final_finding_type,
        model_trained=model_trained,
        severity=final_severity,
        note=data.note
    )

    db.add(review)

    analysis.has_finding = data.has_finding
    analysis.finding_type = final_finding_type
    analysis.result = review_label
    analysis.status = AnalysisStatus.trained if model_trained else AnalysisStatus.reviewed

    db.commit()
    db.refresh(review)
    return review
