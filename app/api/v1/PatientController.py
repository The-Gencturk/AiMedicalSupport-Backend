from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.core.rbac import require_permission
from app.db.DbContext import get_db
from app.schemas.patient import PatientCreate, PatientUpdate, PatientResponse, PatientAllResponse
from app.services.patient_service import (
    create_patient, get_all_patients, get_patient, update_patient, delete_patient
)

router = APIRouter()


@router.post("/", response_model=PatientResponse, dependencies=[Depends(require_permission("user:create"))])
def add_patient(data: PatientCreate, db: Session = Depends(get_db)):
    return create_patient(db, data)


@router.get("/", response_model=list[PatientAllResponse], dependencies=[Depends(require_permission("user:read"))])
def list_patients(db: Session = Depends(get_db)):
    return get_all_patients(db)


@router.get("/{patient_id}", response_model=PatientResponse, dependencies=[Depends(require_permission("user:read"))])
def patient_detail(patient_id: int, db: Session = Depends(get_db)):
    return get_patient(db, patient_id)


@router.put("/{patient_id}", response_model=PatientResponse, dependencies=[Depends(require_permission("user:update"))])
def update_patient_endpoint(patient_id: int, data: PatientUpdate, db: Session = Depends(get_db)):
    return update_patient(db, patient_id, data)


@router.delete("/{patient_id}", dependencies=[Depends(require_permission("user:delete"))])
def delete_patient_endpoint(patient_id: int, db: Session = Depends(get_db)):
    return delete_patient(db, patient_id)