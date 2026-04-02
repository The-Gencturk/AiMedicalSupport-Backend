from fastapi import APIRouter, Depends, UploadFile, File, Form, HTTPException
from sqlalchemy.orm import Session
from app.db.DbContext import get_db
from app.models.OrganModel import OrganModel
from app.services.classification_service import ClassificationService
from app.core.Security import get_current_user
from pathlib import Path

router = APIRouter()

BASE_DIR = Path(__file__).resolve().parents[4]
AI_MODELS_DIR = BASE_DIR / "AiModels"


@router.get("/GetAll")
def get_all_organs(db: Session = Depends(get_db)):
    organs = db.query(OrganModel).all()
    return [
        {
            "id": o.id,
            "name": o.name,
            "display_name": o.display_name,
            "model_path": o.model_path,
            "is_active": o.is_active,
            "created_at": o.created_at,
        }
        for o in organs
    ]


@router.post("/Upload")
async def upload_organ_model(
    name: str = Form(...),
    display_name: str = Form(...),
    model_file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    # Klasörü oluştur
    organ_dir = AI_MODELS_DIR / name.capitalize()
    organ_dir.mkdir(parents=True, exist_ok=True)

    # Dosyayı kaydet
    filename = model_file.filename
    if not filename.endswith((".h5", ".keras")):
        raise HTTPException(status_code=400, detail="Sadece .h5 veya .keras dosyaları kabul edilir.")

    save_path = organ_dir / filename
    content = await model_file.read()
    with open(save_path, "wb") as f:
        f.write(content)

    relative_path = f"AiModels/{name.capitalize()}/{filename}"

    # DB kaydı
    organ = db.query(OrganModel).filter(OrganModel.name == name).first()
    if organ:
        organ.model_path = relative_path
        organ.is_active = True
        organ.uploaded_by = current_user.id
    else:
        organ = OrganModel(
            name=name,
            display_name=display_name,
            model_path=relative_path,
            is_active=True,
            uploaded_by=current_user.id,
        )
        db.add(organ)

    db.commit()
    db.refresh(organ)

    # ClassificationService'e yükle
    classification = ClassificationService()
    classification.reload_organ(organ)

    return {"message": f"{display_name} modeli başarıyla yüklendi.", "organ": organ.name}


@router.patch("/Toggle/{organ_id}")
def toggle_organ(organ_id: int, db: Session = Depends(get_db)):
    organ = db.query(OrganModel).filter(OrganModel.id == organ_id).first()
    if not organ:
        raise HTTPException(status_code=404, detail="Organ bulunamadı.")
    if not organ.model_path:
        raise HTTPException(status_code=400, detail="Bu organ için model yüklenmemiş.")

    organ.is_active = not organ.is_active
    db.commit()

    classification = ClassificationService()
    if organ.is_active:
        classification.reload_organ(organ)
    else:
        classification.remove_organ(organ.name)

    return {"message": f"{organ.display_name} {'aktif' if organ.is_active else 'deaktif'} edildi."}


@router.delete("/DeleteModel/{organ_id}")
async def delete_organ_model(
    organ_id: int,
    db: Session = Depends(get_db),
):
    organ = db.query(OrganModel).filter(OrganModel.id == organ_id).first()
    if not organ:
        raise HTTPException(status_code=404, detail="Organ bulunamadı.")

    if organ.model_path:
        model_path = BASE_DIR / organ.model_path
        if model_path.exists():
            model_path.unlink()

    db.delete(organ)
    db.commit()

    return {"message": f"{organ.display_name} modeli başarıyla silindi."}



@router.post("/analyzePredict")
async def predict_scan_type(
    file: UploadFile = File(...),
):
    image_bytes = await file.read()
    result = ClassificationService().predict_scan_type(image_bytes)  
    return {
        "success": True,
        "data": result
    }