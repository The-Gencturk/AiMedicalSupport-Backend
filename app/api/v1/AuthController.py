from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.db.DbContext import get_db
from app.schemas.auth import UserRegister, UserLogin, UserFullResponse,UpdateMe,MeResponse
from app.services.auth_service import register_user, login_user,update_me_user,update_profile_photo
from app.core.Security import get_current_user
from app.models.User import User
from fastapi import APIRouter, Depends, Response,UploadFile, File
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
        secure=False,          
        samesite="lax",      
        max_age=60 * 60 * 24  # 1 gün
    )
    return {"user": result["user"], "message": "Giriş başarılı"}

@router.get("/MeDetails", response_model=UserFullResponse)
def get_me(current_user: User = Depends(get_current_user)):
    return {
        "id": current_user.id,
        "full_name": current_user.full_name,
        "email": current_user.email,
        "profile" : current_user.profile, 
        "specialty": current_user.specialty,
        "UserProfile":current_user.profile,
        "is_active": current_user.is_active,
        "created_at": current_user.created_at,
        "roles": current_user.get_roles()
    }




@router.get("/Me", response_model=MeResponse)
def get_me(current_user: User = Depends(get_current_user)):
    return {
        "id": current_user.id,
        "full_name": current_user.full_name,
        "email": current_user.email,
        "profile" : current_user.profile, 
        "specialty": current_user.specialty,
        "UserProfile":current_user.profile,
        "is_active": current_user.is_active,
        "roles": current_user.get_roles(),
        "created_at" : current_user.created_at
    }





@router.put("/UpdateMeData", response_model=UserFullResponse)
def update_me(
    data: UpdateMe,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    return update_me_user(db, current_user, data)


    
@router.post("/logout")
def logout(response: Response):
    response.delete_cookie(key="access_token")
    return {"message": "Çıkış başarılı"}



@router.post("/upload-profile")
def upload_profile(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    return update_profile_photo(db, current_user, file)