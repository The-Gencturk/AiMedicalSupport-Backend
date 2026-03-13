import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model, Model

BASE_DIR = r"C:\Users\LENOVO\Desktop\Projeler\AiMedicalSupport-Backend"
MODEL_PATH = os.path.join(BASE_DIR, "AiModels", "beyin_bt_modeli.h5")

class RadiologyService:
    def __init__(self):
        self.model = load_model(MODEL_PATH)
        self.last_conv_layer = self._find_last_conv_layer()

    def _find_last_conv_layer(self):
        for layer in reversed(self.model.layers):
            if 'conv' in layer.name.lower():
                return layer.name
        return None

    def analyze(self, image_bytes: bytes) -> dict:
        # Bytes'dan resmi oku
        np_arr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        # Modele hazırla
        img_resized = cv2.resize(img, (224, 224))
        img_norm = img_resized / 255.0
        img_input = np.reshape(img_norm, (1, 224, 224, 3))

        # Tahmin
        pred = self.model.predict(img_input, verbose=0)[0][0]

        return {
            "result": "KANAMA" if pred > 0.5 else "NORMAL",
            "confidence": int(pred * 100) if pred > 0.5 else int((1 - pred) * 100),
            "is_bleeding": bool(pred > 0.5)
        }
