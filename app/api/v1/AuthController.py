from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.db.DbContext import get_db
from app.schemas.auth import UserRegister, UserLogin, LoginResponse, UserFullResponse
from app.services.auth_service import register_user, login_user
from app.core.Security import get_current_user
from app.models.User import User
from fastapi import APIRouter, Depends, Response
from fastapi.responses import JSONResponse


router = APIRouter()


@router.post("/register", response_model=UserFullResponse, status_code=201)
def register(data: UserRegister, db: Session = Depends(get_db)):
    return register_user(db, data)



@router.post("/login")
def login(data: UserLogin, response: Response, db: Session = Depends(get_db)):
    result = login_user(db, data.email, data.password)
    
    response.set_cookie(
        key="access_token",
        value=result["access_token"],
        httponly=True,       
        secure=True,          
        samesite="lax",      
        max_age=60 * 60 * 24  # 1 gün
    )
    return {"user": result["user"], "message": "Giriş başarılı"}

@router.get("/me", response_model=UserFullResponse)
def get_me(current_user: User = Depends(get_current_user)):
    return {
        "id": current_user.id,
        "full_name": current_user.full_name,
        "email": current_user.email,
        "specialty": current_user.specialty,
        "is_active": current_user.is_active,
        "created_at": current_user.created_at,
        "roles": current_user.get_roles()
    }




@router.post("/logout")
def logout(response: Response):
    response.delete_cookie(key="access_token")
    return {"message": "Çıkış başarılı"}