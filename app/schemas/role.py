from pydantic import BaseModel, EmailStr
from typing import Optional
from datetime import datetime
from typing import List

class UpdateUserRoleSchema(BaseModel):
    id: int
    role_ids: List[int]


class RolePermissionUpdateSchema(BaseModel):
    permission_ids: List[int]


class PermissionResponse(BaseModel):
    id: int
    name: str
    description: Optional[str]

    class Config:
        from_attributes = True


class RoleResponse(BaseModel):
    id: int
    name: str

    class Config:
        from_attributes = True


class RoleWithPermissionsResponse(BaseModel):
    id: int
    name: str
    description: Optional[str]
    permissions: List[PermissionResponse] = []

    class Config:
        from_attributes = True