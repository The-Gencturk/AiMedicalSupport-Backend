from pydantic import BaseModel, EmailStr
from typing import Optional
from datetime import datetime
from typing import Optional, List

class UserRegister(BaseModel):
    full_name: str
    email: EmailStr
    password: str
    specialty: Optional[str] = None


class UserLogin(BaseModel):
    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class UserResponse(BaseModel):
    id: int
    full_name: str
    email: str
    specialty: Optional[str]
    is_active: bool
    created_at: datetime
    roles : List[str] = []

    class Config:
        from_attributes = True


class UserFullResponse(BaseModel):
    id: int
    full_name: str
    email: str
    specialty: Optional[str]
    is_active: bool
    created_at: datetime


    class Config:
        from_attributes = True

class UserMediumRespone(BaseModel):
    id: int
    full_name: str
    email: str
    specialty: Optional[str]
    is_active: bool

    class Config:
        from_attributes = True


class LoginResponse(BaseModel):
    user: UserFullResponse
    message: str


