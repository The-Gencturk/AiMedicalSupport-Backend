from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.core.rbac import require_role, require_permission
from app.core.Security import get_current_user
from app.db.DbContext import get_db
from app.schemas.personel import UserUpdate
from app.services.personel_service import get_all_personel,delete_bypersonel, update_bypersonel

router = APIRouter()


@router.get("/GetAllPersonel", dependencies=[Depends(require_permission("user:read"))])
def getall_personel(db: Session = Depends(get_db)):
    return get_all_personel(db)


@router.delete("/DeletePersonel/{id}", dependencies=[Depends(require_permission("user:delete"))])
def delete_personel(id: int, db: Session = Depends(get_db)):
    return delete_bypersonel(db, id)


@router.put("/UpdatePersonel/{id}", dependencies=[Depends(require_permission("user:update"))])
def update_personel(id: int, data: UserUpdate, db: Session = Depends(get_db)):
    return update_bypersonel(db, id, data)
