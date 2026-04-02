from pathlib import Path
import traceback
import json
import cv2
import numpy as np
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
            cls._instance._classifier = None
            cls._instance._class_indices = None
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

    def load_classifier(self):
        from tensorflow.keras.models import load_model
        classifier_path = BASE_DIR / "AiModels" / "Classifier" / "classifier.keras"
        indices_path = BASE_DIR / "AiModels" / "Classifier" / "class_indices.json"
        try:
            self._classifier = load_model(str(classifier_path), compile=False)
            with open(indices_path) as f:
                raw = json.load(f)
                self._class_indices = {v: k for k, v in raw.items()}
            print(f"[ClassificationService] Classifier yüklendi: {self._class_indices}")
        except Exception as e:
            print(f"[ClassificationService] Classifier yüklenemedi: {e}")

    def predict_scan_type(self, image_bytes: bytes) -> dict:
        if self._classifier is None:
            return {
                "suggested_scan_type": None,
                "confidence": None,
                "auto_detected": False,
                "message": "Classifier yüklenmemiş, manuel seçin.",
                "available_types": self.supported_types
            }
        try:
            np_arr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            img = cv2.resize(img, (224, 224)) / 255.0
            img_input = np.reshape(img, (1, 224, 224, 3))

            preds = self._classifier.predict(img_input, verbose=0)[0]
            idx = int(np.argmax(preds))
            confidence = int(preds[idx] * 100)
            organ = self._class_indices[idx]

            return {
                "suggested_scan_type": organ,
                "confidence": confidence,
                "auto_detected": True,
                "message": f"{organ} tespit edildi.",
                "available_types": self.supported_types
            }
        except Exception as e:
            return {
                "suggested_scan_type": None,
                "confidence": None,
                "auto_detected": False,
                "message": f"Tahmin hatası: {e}",
                "available_types": self.supported_types
            }

    def get_service(self, scan_type: str) -> BaseRadiologyService:
        service = self._services.get(scan_type)
        if not service:
            raise ValueError(f"'{scan_type}' için aktif model bulunamadı.")
        return service

    def reload_organ(self, organ: OrganModel):
        try:
            if organ.name == "brain":
                self._services["brain"] = BrainRadiologyService()
            else:
                model_path = str(BASE_DIR / organ.model_path)
                self._services[organ.name] = GenericRadiologyService(
                    model_path=model_path,
                    organ_name=organ.name,
                )
            print(f"[ClassificationService] Reload edildi: {organ.name}")
        except Exception as e:
            print(f"[ClassificationService] Reload HATA {organ.name}: {e}")
            traceback.print_exc()

    def remove_organ(self, organ_name: str):
        self._services.pop(organ_name, None)

    @property
    def supported_types(self) -> list[str]:
        return list(self._services.keys())