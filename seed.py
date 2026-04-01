from app.models.User import User
from app.db.DbContext import SessionLocal
from app.models.rbac import Role, Permission, RolePermission
from app.models.OrganModel import OrganModel 

def seed_organs(db):
    existing = db.query(OrganModel).first()
    if existing:
        print("Organlar zaten mevcut, atlandı.")
        return

    organs = [
        OrganModel(name="brain",   display_name="Beyin",   model_path="AiModels/Brain/beyin_bt_modeli.keras", is_active=True),
        OrganModel(name="lung",    display_name="Akciğer", model_path=None, is_active=False),
        OrganModel(name="bone",    display_name="Kemik",   model_path=None, is_active=False),
        OrganModel(name="abdomen", display_name="Karın",   model_path=None, is_active=False),
    ]
    db.add_all(organs)
    db.flush()
    print("Organlar eklendi.")

def seed():
    db = SessionLocal()
    try:
        permissions_data = [
            ("analyze:create", "Röntgen analizi oluştur"),
            ("analyze:read",   "Röntgen analizini incele"),
            ("analyze:delete", "Röntgen analizini sil"),
            ("analyze:Update", "Röntgen analizini düzenle"),
            ("analyze:detread", "Röntgen analizini detaylı incele"),
            ("user:read",      "Personelleri listele"),
            ("user:detread",   "Personelleri detaylı incele"),
            ("user:create",    "Personelleri oluştur"),
            ("user:update",    "Personelleri güncelle"),
            ("user:delete",    "Personelleri sil"),
            ("patient:read",   "Hastaları listele"),
            ("patient:detread","Hastaları detaylı incele"),
            ("patient:create", "Hastaları oluştur"),
            ("patient:update", "Hastaları güncelle"),
            ("patient:delete", "Hastaları sil"),
            ("role:manage",    "Rolleri full kotrol"),
            ("role:read",      "Rolleri göster"),
            ("role:detread",   "Rolleri detaylı incele"),
            ("role:update",    "Rolleri düzenle"),
            ("role:delete",    "Rolleri sil"),
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
            "Başhekim Yardımcısı": ["analyze:create","analyze:read","analyze:delete","user:read","user:detread","user:update","patient:read","patient:detread","patient:update"],
            "Klinik Şefi": ["analyze:create","analyze:read","user:read","patient:read","patient:detread","patient:update"],
            "Uzman_Doktor": ["analyze:create","analyze:read","patient:read","patient:detread","patient:update"],
            "Doktor": ["analyze:create","analyze:read","patient:read","patient:detread"],
            "Radyolog": ["analyze:create","analyze:read","analyze:delete","patient:read","patient:detread"],
            "stajer": ["analyze:read","patient:read"],
            "Hasta": ["analyze:read","analyze:create"],
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
        print("Roller ve yetkiler başarıyla oluşturuldu")

        seed_organs(db)  # ← organları ekle
        db.commit()

    except Exception as e:
        db.rollback()
        print(f"❌ Hata: {e}")
    finally:
        db.close()

if __name__ == "__main__":
    seed()