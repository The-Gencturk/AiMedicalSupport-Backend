# app/services/radiology/base_radiology.py
from abc import ABC, abstractmethod
from typing import Optional
import numpy as np

class BaseRadiologyService(ABC):

    @abstractmethod
    def analyze(self, image_bytes: bytes) -> dict:
        """
        Döndürmesi gereken format:
        {
            "result": str,
            "confidence": int,
            "finding": bool,          # genel bulgu var mı (kanama, nodül vs.)
            "finding_type": str|None, # spesifik tür
            "finding_type_confidence": int|None
        }
        """
        pass

    @abstractmethod
    def _get_heatmap(self, img_input: np.ndarray) -> Optional[np.ndarray]:
        pass

    def generate_heatmap(self, img_input: np.ndarray) -> Optional[np.ndarray]:
        """
        Public and backward-compatible heatmap entrypoint.
        New integrations can call this method while legacy services can
        keep implementing `_get_heatmap`.
        """
        return self._get_heatmap(img_input)

    @abstractmethod
    def train(self, image_bytes: bytes, label: int, **kwargs) -> dict:
        pass
