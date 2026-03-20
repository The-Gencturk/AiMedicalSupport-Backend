from sqlalchemy.orm import Session
from app.models.User import User
from app.schemas.personel import AllPersonelResponse, PersonelResponse , UserUpdate
from fastapi import HTTPException


def get_all_personel(db: Session):
    users = db.query(User).all()
    result = []
    for u in users:
        data = AllPersonelResponse(
            id=u.id,
            full_name=u.full_name,
            specialty=u.specialty,
            is_active=u.is_active,
        )
        result.append(data)
    return result


def get_ById_personel(db: Session, user_id: int):
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="Kullanıcı bulunamadı")
    return PersonelResponse(
        id=user.id,
        full_name=user.full_name,
        email=user.email,
        profile = user.profile,
        specialty=user.specialty,
        is_active=user.is_active,
        roles=user.get_roles()
    )


def delete_bypersonel(db: Session, user_id: int):
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="Kullanıcı bulunamadı")
    db.delete(user)
    db.commit()
    return {"message": "Kullanıcı silindi"}


def update_bypersonel(db: Session, user_id: int, data: UserUpdate):
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="Kullanıcı bulunamadı")
    for key, value in data.model_dump(exclude_none=True).items():
        setattr(user, key, value)
    db.commit()
    db.refresh(user)
    return user
