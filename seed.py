from app.models.User import User
from app.db.DbContext import SessionLocal
from app.models.rbac import Role, Permission, RolePermission

def seed():
    db = SessionLocal()
    try:
        # === Yetkiler ===
        permissions_data = [
            ("analyze:create", "Röntgen analizi oluştur"),
            ("analyze:read",   "Röntgen analizini incele"),
            ("analyze:delete", "Röntgen analizini sil"),
            ("user:read",      "Kullanıcıları listele"),
            ("user:create",    "Kullanıcı oluştur"),
            ("user:update",    "Kullanıcı güncelle"),
            ("user:delete",    "Kullanıcı sil"),
            ("role:manage",    "Rol yönetimi"),
        ]

        perms = {}
        for name, desc in permissions_data:
            p = db.query(Permission).filter(Permission.name == name).first()
            if not p:
                p = Permission(name=name, description=desc)
                db.add(p)
                db.flush()
            perms[name] = p


        roles_data = {
            "SuperAdmin": list(perms.keys()),
            "Başhekim": list(perms.keys()),
            "Başhekim Yardımcısı":  ["analyze:create", "analyze:read", "user:read","user:update","user:create","user:delete","analyze:delete","user:update"],
            "Klinik Şefi": ["analyze:create", "analyze:read", "user:read","user:update","user:create","user:delete","analyze:delete","user:update"],
            "Uzman_Doktor": ["analyze:create", "analyze:read", "user:read","user:update","user:create","user:delete","analyze:delete","user:update"],
            "Doktor": ["analyze:create", "analyze:read", "user:read","user:read","user:update","user:create","analyze:delete"],
            "Radyolog": ["analyze:create", "analyze:read", "analyze:delete"],
            "Hasta": ["analyze:read"],
        }

        for role_name, perm_names in roles_data.items():
            role = db.query(Role).filter(Role.name == role_name).first()
            if not role:
                role = Role(name=role_name, description=f"{role_name} rolü")
                db.add(role)
                db.flush()

            for perm_name in perm_names:
                exists = db.query(RolePermission).filter(
                    RolePermission.role_id == role.id,
                    RolePermission.permission_id == perms[perm_name].id
                ).first()
                if not exists:
                    db.add(RolePermission(role_id=role.id, permission_id=perms[perm_name].id))

        db.commit()
        print("✅ Roller ve yetkiler başarıyla oluşturuldu!")

    except Exception as e:
        db.rollback()
        print(f"❌ Hata: {e}")
    finally:
        db.close()

if __name__ == "__main__":
    seed()