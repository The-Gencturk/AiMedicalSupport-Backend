from sqlalchemy.orm import Session
from fastapi import HTTPException, status
from app.models.rbac import UserRole, Role, Permission, RolePermission
from app.models.User import User
from app.schemas.auth import UserRegister
from app.core.Security import hash_password, verify_password, create_access_token
from app.schemas.auth import UserMediumRespone
from app.schemas.role import RoleResponse, PermissionResponse, RoleWithPermissionsResponse,AllPersonel



def update_user_role(db: Session, user_id: int, role_ids: list[int]):
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="Kullanıcı bulunamadı")
    

    db.query(UserRole).filter(UserRole.user_id == user_id).delete()
    
    for role_id in role_ids:
        db.add(UserRole(user_id=user_id, role_id=role_id))
    
    db.commit()
    db.refresh(user)
    return UserMediumRespone.model_validate(user)

def get_all_roles(db: Session) -> list[RoleResponse]:
    roles = db.query(Role).all()
    return [RoleResponse.model_validate(r) for r in roles]


def get_all_permissions(db: Session) -> list[PermissionResponse]:
    perms = db.query(Permission).all()
    return [PermissionResponse.model_validate(p) for p in perms]


def get_role_permissions(db: Session, role_id: int) -> RoleWithPermissionsResponse:
    role = db.query(Role).filter(Role.id == role_id).first()
    if not role:
        raise HTTPException(status_code=404, detail="Rol bulunamadı")
    permissions = [rp.permission for rp in role.permissions]
    return RoleWithPermissionsResponse(
        id=role.id,
        name=role.name,
        description=role.description,
        permissions=[PermissionResponse.model_validate(p) for p in permissions]
    )


def get_user_permissions(db: Session, user_id: int) -> list[PermissionResponse]:
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="Kullanıcı bulunamadı")
    perms = set()
    result = []
    for ur in user.roles:
        for rp in ur.role.permissions:
            if rp.permission.id not in perms:
                perms.add(rp.permission.id)
                result.append(PermissionResponse.model_validate(rp.permission))
    return result


def update_role_permissions(db: Session, role_id: int, permission_ids: list[int]):
    role = db.query(Role).filter(Role.id == role_id).first()
    if not role:
        raise HTTPException(status_code=404, detail="Rol bulunamadı")

    db.query(RolePermission).filter(RolePermission.role_id == role_id).delete()
    for perm_id in permission_ids:
        perm = db.query(Permission).filter(Permission.id == perm_id).first()
        if not perm:
            raise HTTPException(status_code=404, detail=f"Yetki bulunamadı: {perm_id}")
        db.add(RolePermission(role_id=role_id, permission_id=perm_id))

    db.commit()
    return get_role_permissions(db, role_id)



