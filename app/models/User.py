from sqlalchemy import Column, Integer, String, Boolean, DateTime
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from app.db.DbContext import Base


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    full_name = Column(String, nullable=False)
    email = Column(String, unique=True, index=True, nullable=False)
    profile = Column(String, unique=True, index=True, nullable=True)
    hashed_password = Column(String, nullable=False)
    specialty = Column(String, nullable=True)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # RBAC ilişkisi
    roles = relationship("UserRole", back_populates="user", cascade="all, delete-orphan")

    def has_role(self, role_name: str) -> bool:
        return any(ur.role.name == role_name for ur in self.roles if ur.role.is_active)

    def has_permission(self, permission_name: str) -> bool:
        for user_role in self.roles:
            if not user_role.role.is_active:
                continue
            for rp in user_role.role.permissions:
                if rp.permission.name == permission_name:
                    return True
        return False

    def get_roles(self) -> list:
        return [ur.role.name for ur in self.roles if ur.role.is_active]

    def get_permissions(self) -> list:
        perms = set()
        for user_role in self.roles:
            if not user_role.role.is_active:
                continue
            for rp in user_role.role.permissions:
                perms.add(rp.permission.name)
        return list(perms)
    
