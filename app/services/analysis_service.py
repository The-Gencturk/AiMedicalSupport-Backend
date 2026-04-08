import os
import cv2
import shutil
import numpy as np
import traceback
from typing import Any, List, Optional
from sqlalchemy.orm import Session
from fastapi import HTTPException

from app.models.AnalisyModel import Analysis, AnalysisReview, AnalysisStatus
from app.schemas.analisy import ReviewCreate, Severity
from app.services.classification_service import ClassificationService

UPLOAD_DIR = r"C:\Users\LENOVO\Desktop\Projeler\AiMedicalSupport-Backend\uploads"
HEATMAP_DIR = r"C:\Users\LENOVO\Desktop\Projeler\AiMedicalSupport-Backend\uploads\heatmaps"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(HEATMAP_DIR, exist_ok=True)


def _resolve_model_input_shape(radiology) -> tuple[int, int, int]:
    model = getattr(radiology, "model", None)
    shape = getattr(model, "input_shape", None)
    if isinstance(shape, list) and shape:
        shape = shape[0]
    if not shape or len(shape) < 4:
        return 224, 224, 3

    h = int(shape[1]) if shape[1] else 224
    w = int(shape[2]) if shape[2] else 224
    c = int(shape[3]) if shape[3] else 3
    return h, w, c


def _prepare_heatmap_input(img_bgr: np.ndarray, radiology) -> np.ndarray:
    h, w, c = _resolve_model_input_shape(radiology)
    resized = cv2.resize(img_bgr, (w, h))

    if c == 1:
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        norm = (gray / 255.0).astype(np.float32)
        return np.reshape(norm, (1, h, w, 1))

    norm = (resized / 255.0).astype(np.float32)
    if c == 3:
        return np.reshape(norm, (1, h, w, 3))

    if c > 3:
        extras = np.repeat(norm[:, :, :1], c - 3, axis=2)
        norm = np.concatenate([norm, extras], axis=2)
    else:
        norm = norm[:, :, :c]

    return np.reshape(norm, (1, h, w, c))


def _run_heatmap_generator(radiology, img_input: np.ndarray):
    for method_name in ("generate_heatmap", "get_heatmap", "_get_heatmap"):
        method = getattr(radiology, method_name, None)
        if not callable(method):
            continue
        try:
            return method(img_input)
        except TypeError:
            try:
                return method(img_input=img_input)
            except TypeError:
                continue
    return None


def _normalize_heatmap_array(heatmap) -> Optional[np.ndarray]:
    if heatmap is None:
        return None

    arr = np.asarray(heatmap, dtype=np.float32)
    if arr.size == 0:
        return None

    arr = np.squeeze(arr)
    if arr.ndim == 3:
        arr = np.mean(arr, axis=-1)
    if arr.ndim != 2:
        return None

    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    min_val = float(np.min(arr))
    max_val = float(np.max(arr))
    if max_val <= min_val:
        return None

    return (arr - min_val) / (max_val - min_val)


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
        return f"ANOMALI ({finding_type})"
    return "ANOMALI"


def _serialize_review_response(review: AnalysisReview, training: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": review.id,
        "analysis_id": review.analysis_id,
        "doctor_id": review.doctor_id,
        "label": review.label,
        "has_finding": review.has_finding,
        "finding_type": review.finding_type,
        "model_trained": review.model_trained,
        "severity": review.severity.value if review.severity else None,
        "note": review.note,
        "created_at": review.created_at,
        "training": training,
    }


def get_all_analyses(db: Session) -> list[Analysis]:
    return db.query(Analysis).order_by(Analysis.created_at.desc()).all()


def get_analysis(db: Session, analysis_id: int) -> Analysis:
    analysis = db.query(Analysis).filter(Analysis.id == analysis_id).first()
    if not analysis:
        raise HTTPException(status_code=404, detail="Analiz bulunamadi")
    return analysis


def delete_analysis(db: Session, analysis_id: int):
    analysis = get_analysis(db, analysis_id)
    analiz_dir = os.path.join(UPLOAD_DIR, f"analiz_{analysis_id}")
    if os.path.exists(analiz_dir):
        shutil.rmtree(analiz_dir, ignore_errors=True)
    db.delete(analysis)
    db.commit()
    return {"message": "Analiz silindi"}


def create_analysis(
    db: Session,
    patient_id: int,
    doctor_id: int,
    image_bytes: bytes,
    filename: str,
    scan_type: str = "brain",
) -> Analysis:
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
        scan_type=scan_type,
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
            raise ValueError("cv2.imdecode basarisiz")

        img_input = _prepare_heatmap_input(img, radiology)
        raw_heatmap = _run_heatmap_generator(radiology, img_input)
        heatmap = _normalize_heatmap_array(raw_heatmap)

        if heatmap is not None:
            heatmap_filename = f"heatmap_{filename}"
            heatmap_disk_path = os.path.join(analiz_dir, heatmap_filename)
            heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
            heatmap_uint8 = np.uint8(255 * heatmap_resized)
            heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
            overlay = cv2.addWeighted(heatmap_colored, 0.4, img, 0.6, 0)

            if cv2.imwrite(heatmap_disk_path, overlay):
                heatmap_url = f"/uploads/analiz_{analysis.id}/{heatmap_filename}"
            else:
                print(f"HEATMAP HATA: dosya kaydedilemedi -> {heatmap_disk_path}")
        else:
            print(f"HEATMAP UYARI: heatmap olusturulamadi, scan_type={scan_type}")

    except Exception as e:
        print(f"HEATMAP HATA: {e}")
        traceback.print_exc()

    analysis.image_path = f"/uploads/analiz_{analysis.id}/{filename}"
    analysis.heatmap_path = heatmap_url
    db.commit()
    db.refresh(analysis)
    return analysis


def add_review(db: Session, analysis_id: int, doctor_id: int, data: ReviewCreate) -> dict[str, Any]:
    analysis = get_analysis(db, analysis_id)
    radiology = ClassificationService().get_service(analysis.scan_type or "brain")

    if not data.has_finding:
        final_finding_type = None
        final_severity = Severity.none
    else:
        final_finding_type = data.finding_type if data.finding_type else None
        final_severity = data.severity
        if not final_finding_type:
            raise HTTPException(status_code=400, detail="Anomali durumunda tur belirtilmelidir.")

    review_label = _build_review_label(
        has_finding=data.has_finding,
        finding_type=final_finding_type,
        label=data.label,
    )

    model_trained = False
    training_result: dict[str, Any] = {
        "attempted": False,
        "success": False,
        "message": "Egitim baslatilmadi.",
    }
    image_disk_path = _resolve_upload_path(analysis.image_path)

    if os.path.exists(image_disk_path):
        with open(image_disk_path, "rb") as f:
            image_bytes = f.read()

        training_result["attempted"] = True
        train_result = radiology.train(
            image_bytes=image_bytes,
            label=1 if data.has_finding else 0,
            finding_type=final_finding_type,
            bleeding_type=final_finding_type,
            analysis_id=analysis_id,
            doctor_id=doctor_id,
        )
        training_result.update(train_result)
        model_trained = bool(train_result.get("success", False))
        if not model_trained:
            print(f"MODEL TRAIN FAILED id={analysis_id}: {train_result.get('message')}")
    else:
        training_result.update(
            {
                "attempted": False,
                "message": f"Egitim atlandi: orijinal goruntu bulunamadi ({analysis.image_path})",
            }
        )

    review = AnalysisReview(
        analysis_id=analysis_id,
        doctor_id=doctor_id,
        label=review_label,
        has_finding=data.has_finding,
        finding_type=final_finding_type,
        model_trained=model_trained,
        severity=final_severity,
        note=data.note,
    )

    db.add(review)

    analysis.has_finding = data.has_finding
    analysis.finding_type = final_finding_type
    analysis.result = review_label
    analysis.status = AnalysisStatus.trained if model_trained else AnalysisStatus.reviewed

    db.commit()
    db.refresh(review)
    return _serialize_review_response(review, training_result)


def bulk_delete_analyses(db: Session, analysis_ids: List[int]) -> dict:
    if not analysis_ids:
        raise HTTPException(status_code=400, detail="En az bir analiz ID'si gerekli.")

    analyses = db.query(Analysis).filter(Analysis.id.in_(analysis_ids)).all()
    found_ids = {a.id for a in analyses}
    not_found = [i for i in analysis_ids if i not in found_ids]

    for analysis_id in found_ids:
        analiz_dir = os.path.join(UPLOAD_DIR, f"analiz_{analysis_id}")
        if os.path.exists(analiz_dir):
            shutil.rmtree(analiz_dir, ignore_errors=True)

    db.query(Analysis).filter(Analysis.id.in_(found_ids)).delete(synchronize_session=False)
    db.commit()

    return {
        "deleted": list(found_ids),
        "not_found": not_found,
        "deleted_count": len(found_ids),
    }
