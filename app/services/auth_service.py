from sqlalchemy.orm import Session
from fastapi import HTTPException, status
from app.models.rbac import Role, UserRole
from app.models.User import User
from app.schemas.auth import UserRegister, UpdateMe
from app.core.Security import hash_password, verify_password, create_access_token
import shutil
import uuid
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
USER_PROFILE_DIR = BASE_DIR / "wwwroot" / "UserProfile"
USER_PROFILE_URL_PREFIX = "/wwwroot/UserProfile"
ALLOWED_IMAGE_TYPES = {
    "image/jpeg": "jpg",
    "image/jpg": "jpg",
    "image/png": "png",
}


def _resolve_profile_disk_path(profile_url: str | None) -> Path | None:
    if not profile_url:
        return None
    filename = Path(profile_url).name
    if not filename:
        return None
    return USER_PROFILE_DIR / filename


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
    # 1. Klasörün varlığından emin ol (Yoksa oluştur)
    upload_dir = "wwwroot/UserProfile"
    if not os.path.exists(upload_dir):
        os.makedirs(upload_dir, exist_ok=True)

    # 2. Eski fotoğrafı sil (Temizlik)
    if user.profile:
        # DB'de "/wwwroot/..." diye saklıyorsan baştaki slash'ı temizle
        old_path = user.profile.lstrip("/") 
        if os.path.exists(old_path):
            try:
                os.remove(old_path)
            except Exception as e:
                print(f"Eski dosya silinemedi: {e}")

    # 3. Dosya tipi kontrolü
    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(400, "Sadece resim (JPEG/PNG) yükleyebilirsiniz")

    # 4. Benzersiz dosya adı oluştur
    ext = file.filename.split(".")[-1]
    filename = f"{uuid.uuid4()}.{ext}"
    filepath = os.path.join(upload_dir, filename)

    # 5. Dosyayı kaydet
    try:
        with open(filepath, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(500, f"Dosya kaydedilirken hata oluştu: {str(e)}")

    # 6. Veritabanını güncelle
    # Başına / ekleyerek kaydediyoruz ki frontend'den erişim kolay olsun
    user.profile = f"/{filepath.replace(os.sep, '/')}" 
    db.commit()
    db.refresh(user)

    return user
