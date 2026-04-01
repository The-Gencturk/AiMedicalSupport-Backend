from app.services.radiology.base_radiology import BaseRadiologyService
import numpy as np

class LungRadiologyService(BaseRadiologyService):
    def analyze(self, image_bytes: bytes) -> dict:
        raise NotImplementedError("Akciğer modeli henüz eklenmedi.")
    
    def _get_heatmap(self, img_input: np.ndarray):
        return None
    
    def train(self, image_bytes: bytes, label: int, **kwargs) -> dict:
        raise NotImplementedError("Akciğer modeli henüz eklenmedi.")