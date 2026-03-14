from fastapi import APIRouter, Depends
from app.core.rbac import require_role, require_permission
from app.core.Security import get_current_user
from app.models.User import User

router = APIRouter()


@router.delete("/user/{id}", dependencies=[Depends(require_role("SuperAdmin","Başhekim"))])
def delete_user(id: int):
    pass


@router.get("/GetAllPersonel", dependencies=[Depends(require_permission("user:read"))])
def list_users():
    pass


# @router.post("/analyze", dependencies=[Depends(require_permission("analyze:create"))])
# def analyze(file: UploadFile):
#     pass


@router.get("/profile")
def profile(current_user: User = Depends(get_current_user)):
    if current_user.has_role("SuperAdmin"):
        return {"user": current_user, "extra": "admin_data"}
    return {"user": current_user}