from sqlalchemy.orm import Session
from fastapi import HTTPException, status
from app.models.rbac import Role, UserRole
from app.models.User import User
from app.schemas.auth import UserRegister, UpdateMe
from app.core.Security import hash_password, verify_password, create_access_token
import os
import shutil
import uuid


def register_user(db: Session, data: UserRegister) -> User:
    existing = db.query(User).filter(User.email == data.email).first()
    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Bu email zaten kayıtlı"
        )

    user = User(
        full_name=data.full_name,
        email=data.email,
        hashed_password=hash_password(data.password),
        specialty="Hasta",
    )
    db.add(user)
    db.flush()

  
    hasta_role = db.query(Role).filter(Role.name == "Hasta").first()
    if hasta_role:
     db.add(UserRole(user_id=user.id, role_id=hasta_role.id))

    db.commit()
    db.refresh(user)
    return user



def update_me_user(db: Session, user: User, data: UpdateMe):

    if data.full_name is not None:
        user.full_name = data.full_name

    if data.specialty is not None:
        user.specialty = data.specialty

    if data.profile is not None:
        user.profile = data.profile

    db.commit()
    db.refresh(user)

    return user


def login_user(db: Session, email: str, password: str) -> dict:
    user = db.query(User).filter(User.email == email).first()

    if not user or not verify_password(password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Email veya şifre hatalı"
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Hesabınız aktif değil"
        )

    token = create_access_token(data={"sub": user.email})
    return {"access_token": token, "token_type": "bearer", "user": user}



def update_profile_photo(db, user, file):

    if user.profile:
        old_path = user.profile.lstrip("/")  
        if os.path.exists(old_path):
            os.remove(old_path)

    if file.content_type not in ["image/jpeg", "image/png"]: raise HTTPException(400, "Sadece resim yükleyebilirsiniz")
  

    ext = file.filename.split(".")[-1]
    filename = f"{uuid.uuid4()}.{ext}"
    filepath = f"wwwroot/UserProfile/{filename}"

  
    with open(filepath, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    user.profile = f"/wwwroot/UserProfile/{filename}"
    db.commit()
    db.refresh(user)

    return user

