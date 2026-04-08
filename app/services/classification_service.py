import json
import traceback
from pathlib import Path

import cv2
import numpy as np
from sqlalchemy.orm import Session

from app.models.OrganModel import OrganModel
from app.services.radiology.base_radiology import BaseRadiologyService
from app.services.radiology.brain_service import BrainRadiologyService
from app.services.radiology.generic_service import GenericRadiologyService

BASE_DIR = Path(__file__).resolve().parents[2]
MIN_AUTO_DETECT_CONFIDENCE = 65
MIN_TOP_GAP_PERCENT = 8


class ClassificationService:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._services = {}
            cls._instance._classifier = None
            cls._instance._class_indices = None
            cls._instance._class_priors = None
        return cls._instance

    @staticmethod
    def _normalize(name: str) -> str:
        return name.strip().lower()

    def load_from_db(self, db: Session):
        organs = db.query(OrganModel).filter_by(is_active=True).all()
        for organ in organs:
            key = self._normalize(organ.name)
            if key in self._services:
                continue
            try:
                if key == "brain":
                    self._services[key] = BrainRadiologyService()
                else:
                    model_path = str(BASE_DIR / organ.model_path)
                    self._services[key] = GenericRadiologyService(
                        model_path=model_path,
                        organ_name=key,
                    )
                print(f"[ClassificationService] Yuklendi: {key}")
            except Exception as e:
                print(f"[ClassificationService] HATA {key}: {e}")
                traceback.print_exc()

    def _build_priors_from_dataset(self, dataset_dir: Path) -> dict[int, float] | None:
        if self._class_indices is None or not dataset_dir.exists():
            return None

        counts: dict[int, int] = {}
        total = 0
        for idx, label in self._class_indices.items():
            class_dir = dataset_dir / label
            if not class_dir.exists() or not class_dir.is_dir():
                continue
            count = 0
            for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"):
                count += len(list(class_dir.glob(ext)))
            if count <= 0:
                continue
            counts[idx] = count
            total += count

        if total <= 0:
            return None
        return {idx: (count / total) for idx, count in counts.items()}

    def _load_class_priors(self, indices_path: Path):
        self._class_priors = None
        if self._class_indices is None:
            return

        priors_path = indices_path.with_name("class_priors.json")
        if priors_path.exists():
            try:
                with open(priors_path, encoding="utf-8") as f:
                    raw = json.load(f)
                if raw and isinstance(next(iter(raw.keys())), str) and not next(iter(raw.keys())).isdigit():
                    # {"brain": 0.2, "lung": 0.6}
                    mapped = {}
                    label_to_idx = {label: idx for idx, label in self._class_indices.items()}
                    for label, prior in raw.items():
                        idx = label_to_idx.get(self._normalize(label))
                        if idx is None:
                            idx = label_to_idx.get(label)
                        if idx is not None:
                            mapped[idx] = float(prior)
                    self._class_priors = mapped or None
                else:
                    # {"0": 0.2, "1": 0.3}
                    self._class_priors = {int(k): float(v) for k, v in raw.items()}
            except Exception as e:
                print(f"[ClassificationService] class_priors okunamadi: {e}")

        if self._class_priors is None:
            dataset_dir = indices_path.parent / "dataset"
            self._class_priors = self._build_priors_from_dataset(dataset_dir)

        if self._class_priors:
            print(f"[ClassificationService] Class priors: {self._class_priors}")

    def _apply_prior_calibration(self, probs: np.ndarray) -> np.ndarray:
        adjusted = np.array(probs, dtype=np.float32)
        if not self._class_priors:
            return adjusted

        for idx, prior in self._class_priors.items():
            if 0 <= idx < len(adjusted):
                adjusted[idx] = adjusted[idx] / max(float(prior), 1e-6)

        total = float(np.sum(adjusted))
        if total > 0:
            adjusted = adjusted / total
        return adjusted

    def load_classifier(self):
        from tensorflow.keras.models import load_model

        classifier_path = BASE_DIR / "AiModels" / "Classifier" / "classifier.keras"
        indices_path = BASE_DIR / "AiModels" / "Classifier" / "class_indices.json"
        try:
            self._classifier = load_model(str(classifier_path), compile=False)
            with open(indices_path, encoding="utf-8") as f:
                raw = json.load(f)
                # {"abdomen": 0, "brain": 1, "lung": 2} -> {0: "abdomen", 1: "brain", 2: "lung"}
                self._class_indices = {int(v): self._normalize(k) for k, v in raw.items()}

            self._load_class_priors(indices_path)
            print(f"[ClassificationService] Classifier yuklendi: {self._class_indices}")

            dummy = np.zeros((1, 224, 224, 3), dtype=np.float32)
            self._classifier.predict(dummy, verbose=0)
            print("[ClassificationService] Classifier warm-up tamamlandi.")
        except Exception as e:
            print(f"[ClassificationService] Classifier yuklenemedi: {e}")

    def predict_scan_type(self, image_bytes: bytes) -> dict:
        if self._classifier is None:
            return {
                "suggested_scan_type": None,
                "confidence": None,
                "auto_detected": False,
                "needs_manual_review": True,
                "message": "Classifier yuklenmemis, manuel secin.",
                "available_types": self.supported_types,
                "candidates": [],
            }

        try:
            np_arr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("Goruntu okunamadi.")

            img = cv2.resize(img, (224, 224)) / 255.0
            img_input = np.reshape(img, (1, 224, 224, 3)).astype(np.float32)

            raw_preds = np.asarray(self._classifier.predict(img_input, verbose=0)[0], dtype=np.float32)
            calibrated_preds = self._apply_prior_calibration(raw_preds)

            available_types = self.supported_types
            available_index_map = {
                idx: label
                for idx, label in self._class_indices.items()
                if label in self._services
            }

            decision_scores = np.array(calibrated_preds, dtype=np.float32)
            if available_index_map:
                mask = np.zeros_like(decision_scores, dtype=np.float32)
                for idx in available_index_map:
                    if 0 <= idx < len(mask):
                        mask[idx] = 1.0
                decision_scores = decision_scores * mask
                masked_total = float(np.sum(decision_scores))
                if masked_total > 0:
                    decision_scores = decision_scores / masked_total
                else:
                    decision_scores = np.array(calibrated_preds, dtype=np.float32)

            sorted_indices = np.argsort(decision_scores)[::-1]
            top_idx = int(sorted_indices[0])
            second_idx = int(sorted_indices[1]) if len(sorted_indices) > 1 else top_idx

            confidence = int(float(decision_scores[top_idx]) * 100)
            top_gap = int((float(decision_scores[top_idx]) - float(decision_scores[second_idx])) * 100)
            organ = self._normalize(self._class_indices[top_idx])
            service_available = organ in self._services

            candidate_indices = sorted_indices
            if available_index_map:
                candidate_indices = np.array(
                    [idx for idx in sorted_indices if int(idx) in available_index_map],
                    dtype=np.int64,
                )
                if len(candidate_indices) == 0:
                    candidate_indices = sorted_indices

            candidates = []
            for idx in candidate_indices[:3]:
                idx = int(idx)
                label = self._normalize(self._class_indices.get(idx, "unknown"))
                candidates.append(
                    {
                        "scan_type": label,
                        "calibrated_confidence": int(float(decision_scores[idx]) * 100),
                        "raw_confidence": int(float(raw_preds[idx]) * 100),
                        "service_available": label in self._services,
                    }
                )

            auto_detected = bool(
                service_available
                and confidence >= MIN_AUTO_DETECT_CONFIDENCE
                and top_gap >= MIN_TOP_GAP_PERCENT
            )

            if auto_detected:
                message = f"{organ} tespit edildi."
            else:
                message = (
                    "Tahmin belirsiz manuel secim onerilir."
                )

            return {
                "suggested_scan_type": organ,
                "confidence": confidence,
                "raw_confidence": int(float(raw_preds[top_idx]) * 100),
                "top_gap": top_gap,
                "auto_detected": auto_detected,
                "needs_manual_review": not auto_detected,
                "service_available": service_available,
                "message": message,
                "available_types": available_types,
                "candidates": candidates,
            }
        except Exception as e:
            return {
                "suggested_scan_type": None,
                "confidence": None,
                "auto_detected": False,
                "needs_manual_review": True,
                "message": f"Tahmin hatasi: {e}",
                "available_types": self.supported_types,
                "candidates": [],
            }

    def get_service(self, scan_type: str) -> BaseRadiologyService:
        key = self._normalize(scan_type)
        service = self._services.get(key)
        if not service:
            raise ValueError(f"'{scan_type}' icin aktif model bulunamadi. Yuklu modeller: {list(self._services.keys())}")
        return service

    def reload_organ(self, organ: OrganModel):
        key = self._normalize(organ.name)
        try:
            if key == "brain":
                self._services[key] = BrainRadiologyService()
            else:
                model_path = str(BASE_DIR / organ.model_path)
                self._services[key] = GenericRadiologyService(
                    model_path=model_path,
                    organ_name=key,
                )
            print(f"[ClassificationService] Reload edildi: {key}")
        except Exception as e:
            print(f"[ClassificationService] Reload HATA {key}: {e}")
            traceback.print_exc()

    def remove_organ(self, organ_name: str):
        self._services.pop(self._normalize(organ_name), None)

    @property
    def supported_types(self) -> list[str]:
        return list(self._services.keys())
