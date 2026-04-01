from sqlalchemy import Column, Integer, String, Boolean, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.db.DbContext import Base

class OrganModel(Base):
    __tablename__ = "organ_models"

    id           = Column(Integer, primary_key=True, index=True)
    name         = Column(String, unique=True, nullable=False)      # "brain", "lung"
    display_name = Column(String, nullable=False)                   # "Beyin", "Akciğer"
    model_path   = Column(String, nullable=True)                    # .h5 dosya yolu
    is_active    = Column(Boolean, default=False, nullable=False)   # model yüklenince true
    uploaded_by  = Column(Integer, ForeignKey("users.id"), nullable=True)
    created_at   = Column(DateTime(timezone=True), server_default=func.now())
    updated_at   = Column(DateTime(timezone=True), onupdate=func.now())

    uploader = relationship("User", foreign_keys=[uploaded_by])