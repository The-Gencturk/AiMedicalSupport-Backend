from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.core.rbac import require_role, require_permission
from app.core.Security import get_current_user
from app.db.DbContext import get_db
from app.models.User import User
from app.schemas.role import UpdateUserRoleSchema, RolePermissionUpdateSchema
from app.services.role_service import (
    update_user_role as role_update,
    get_all_roles,
    get_all_permissions,
    get_role_permissions,
    get_user_permissions,
    update_role_permissions
)

router = APIRouter()


@router.delete("/user/{id}", dependencies=[Depends(require_role("SuperAdmin", "Başhekim"))])
def delete_user(id: int):
    pass





@router.put("/UpdateUserRole", dependencies=[Depends(require_permission("role:manage"))])
def update_user_role(data: UpdateUserRoleSchema, db: Session = Depends(get_db)):
    return role_update(db, data.id, data.role_ids)


@router.get("/AllRoles", dependencies=[Depends(require_permission("role:manage"))])
def all_roles(db: Session = Depends(get_db)):
    return get_all_roles(db)


@router.get("/AllPermissions", dependencies=[Depends(require_permission("role:manage"))])
def all_permissions(db: Session = Depends(get_db)):
    return get_all_permissions(db)


@router.get("/RolePermissions/{role_id}", dependencies=[Depends(require_permission("role:manage"))])
def role_permissions(role_id: int, db: Session = Depends(get_db)):
    return get_role_permissions(db, role_id)


@router.get("/UserPermissions/{user_id}", dependencies=[Depends(require_permission("role:manage"))])
def user_permissions(user_id: int, db: Session = Depends(get_db)):
    return get_user_permissions(db, user_id)


@router.put("/UpdateRolePermissions/{role_id}", dependencies=[Depends(require_permission("role:manage"))])
def update_role_perms(role_id: int, data: RolePermissionUpdateSchema, db: Session = Depends(get_db)):
    return update_role_permissions(db, role_id, data.permission_ids)



@router.get("/profile")
def profile(current_user: User = Depends(get_current_user)):
    return {
        "id": current_user.id,
        "full_name": current_user.full_name,
        "email": current_user.email,
        "specialty": current_user.specialty,
        "is_active": current_user.is_active,
        "created_at": current_user.created_at,
        "roles": current_user.get_roles()
    }

