from pydantic import BaseModel, EmailStr
from typing import Optional
from datetime import date, datetime
from enum import Enum

class Gender(str, Enum):
    male = "erkek"
    female = "kadın"
    other = "diğer"

class BloodType(str, Enum):
    a_pos = "A+"
    a_neg = "A-"
    b_pos = "B+"
    b_neg = "B-"
    ab_pos = "AB+"
    ab_neg = "AB-"
    o_pos = "0+"
    o_neg = "0-"

class PatientCreate(BaseModel):
    full_name: str
    tc_no: Optional[str] = None
    birth_date: Optional[date] = None
    gender: Optional[Gender] = None
    blood_type: Optional[BloodType] = None
    phone: Optional[str] = None
    email: Optional[str] = None
    address: Optional[str] = None
    medical_history: Optional[str] = None
    allergies: Optional[str] = None
    medications: Optional[str] = None
    notes: Optional[str] = None

class PatientUpdate(PatientCreate):
    full_name: Optional[str] = None

class PatientResponse(BaseModel):
    id: int
    full_name: str
    tc_no: Optional[str]
    birth_date: Optional[date]
    gender: Optional[Gender]
    blood_type: Optional[BloodType]
    phone: Optional[str]
    email: Optional[str]
    address: Optional[str]
    medical_history: Optional[str]
    allergies: Optional[str]
    medications: Optional[str]
    notes: Optional[str]
    is_active: bool
    created_at: datetime

    class Config:
        from_attributes = True

class PatientAllResponse(BaseModel):
    id: int
    full_name: str
    birth_date: Optional[date]
    gender: Optional[Gender]
    notes: Optional[str]
    is_active: bool
    created_at: datetime

    class Config:
        from_attributes = True

