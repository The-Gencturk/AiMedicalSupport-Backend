from pathlib import Path
import traceback
from sqlalchemy.orm import Session
from app.services.radiology.base_radiology import BaseRadiologyService
from app.services.radiology.brain_service import BrainRadiologyService
from app.services.radiology.generic_service import GenericRadiologyService
from app.models.OrganModel import OrganModel

BASE_DIR = Path(__file__).resolve().parents[2]


class ClassificationService:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._services = {}
        return cls._instance

    def load_from_db(self, db: Session):
        organs = db.query(OrganModel).filter_by(is_active=True).all()
        for organ in organs:
            if organ.name in self._services:
                continue
            try:
                if organ.name == "brain":
                    self._services["brain"] = BrainRadiologyService()
                else:
                    model_path = str(BASE_DIR / organ.model_path)
                    self._services[organ.name] = GenericRadiologyService(
                        model_path=model_path,
                        organ_name=organ.name,
                    )
                print(f"[ClassificationService] Yüklendi: {organ.name}")
            except Exception as e:
                print(f"[ClassificationService] HATA {organ.name}: {e}")
                traceback.print_exc()

    def get_service(self, scan_type: str) -> BaseRadiologyService:
        service = self._services.get(scan_type)
        if not service:
            raise ValueError(f"'{scan_type}' için aktif model bulunamadı.")
        return service

    def reload_organ(self, organ: OrganModel):
        """Yeni model yüklenince sadece o organı reload et."""
        try:
            model_path = str(BASE_DIR / organ.model_path)
            if organ.name == "brain":
                self._services["brain"] = BrainRadiologyService()
            else:
                self._services[organ.name] = GenericRadiologyService(
                    model_path=model_path,
                    organ_name=organ.name,
                )
            print(f"[ClassificationService] Reload edildi: {organ.name}")
        except Exception as e:
            print(f"[ClassificationService] Reload HATA {organ.name}: {e}")

    def remove_organ(self, organ_name: str):
        """Organ deaktif edilince bellekten kaldır."""
        self._services.pop(organ_name, None)

    @property
    def supported_types(self) -> list[str]:
        return list(self._services.keys())