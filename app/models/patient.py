from sqlalchemy import Column, Integer, String, DateTime, Date, Enum, Text, Boolean
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.db.DbContext import Base
import enum

class Gender(str, enum.Enum):
    male = "erkek"
    female = "kadın"
    other = "diğer"

class BloodType(str, enum.Enum):
    a_pos = "A+"
    a_neg = "A-"
    b_pos = "B+"
    b_neg = "B-"
    ab_pos = "AB+"
    ab_neg = "AB-"
    o_pos = "0+"
    o_neg = "0-"

class Patient(Base):
    __tablename__ = "patients"

    id = Column(Integer, primary_key=True, index=True)
    full_name = Column(String, nullable=False)
    tc_no = Column(String, unique=True, nullable=True)
    birth_date = Column(Date, nullable=True)
    gender = Column(Enum(Gender), nullable=True)
    blood_type = Column(Enum(BloodType), nullable=True)
    phone = Column(String, nullable=True)
    email = Column(String, nullable=True)
    address = Column(Text, nullable=True)
    medical_history = Column(Text, nullable=True)  
    allergies = Column(Text, nullable=True)     
    medications = Column(Text, nullable=True)       
    notes = Column(Text, nullable=True)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    analyses = relationship("Analysis", back_populates="patient", cascade="all, delete-orphan")