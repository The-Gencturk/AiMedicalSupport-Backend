from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime


class PersonelResponse(BaseModel):
    id: int
    full_name: str
    email: str
    specialty: Optional[str]
    is_active: bool
    roles: List[str] = []
    
    class Config:
        from_attributes = True


class UserUpdate(BaseModel):
    full_name: Optional[str] = None
    specialty: Optional[str] = None
    is_active: Optional[bool] = None