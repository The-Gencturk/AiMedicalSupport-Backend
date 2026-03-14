from fastapi import Depends, HTTPException, status
from app.core.Security import get_current_user
from app.models.User import User


def require_role(*role_names: str):
    """
    Kullanım:
        @router.get("/admin", dependencies=[Depends(require_role("SuperAdmin"))])
    """
    def checker(current_user: User = Depends(get_current_user)):
        for role in role_names:
            if current_user.has_role(role):
                return current_user
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Bu işlem için yetkiniz yok. Gerekli roller: {', '.join(role_names)}"
        )
    return checker


def require_permission(*permission_names: str):
    """
    Kullanım:
        @router.post("/analyze", dependencies=[Depends(require_permission("analyze:create"))])
    """
    def checker(current_user: User = Depends(get_current_user)):
        for perm in permission_names:
            if current_user.has_permission(perm):
                return current_user
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Bu işlem için yetkiniz yok. Gerekli izinler: {', '.join(permission_names)}"
        )
    return checker