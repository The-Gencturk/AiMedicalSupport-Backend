from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime


class AllPersonelResponse(BaseModel):
    id: int
    full_name: str
    specialty: Optional[str]
    is_active: bool
    email: str
    
    class Config:
        from_attributes = True



class PersonelResponse(BaseModel):
    id: int
    full_name: str
    email: str
    profile: Optional[str] = None 
    specialty: Optional[str]
    is_active: bool
    roles: List[str] = []
    
    class Config:
        from_attributes = True


class UserUpdate(BaseModel):
    full_name: Optional[str] = None
    email: Optional[str] = None
    specialty: Optional[str] = None
    is_active: Optional[bool] = None