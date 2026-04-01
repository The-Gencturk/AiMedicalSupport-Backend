import threading
from pathlib import Path
from typing import Optional
import cv2
import numpy as np
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from app.services.radiology.base_radiology import BaseRadiologyService

IMAGE_SIZE = (224, 224)
FINDING_THRESHOLD = 0.5


class GenericRadiologyService(BaseRadiologyService):

    def __init__(self, model_path: str, organ_name: str):
        self.organ_name = organ_name
        self._model_lock = threading.RLock()
        self.model = load_model(model_path, compile=False)
        self.model.compile(
            optimizer=Adam(learning_rate=0.00001),
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )
        self.last_conv_layer = self._find_last_conv_layer()

    def _find_last_conv_layer(self) -> Optional[str]:
        for layer in reversed(self.model.layers):
            if "conv" in layer.name.lower():
                return layer.name
        return None

    def _decode_image(self, image_bytes: bytes) -> np.ndarray:
        np_arr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Image decode failed.")
        return img

    def _prepare_input(self, bgr_img: np.ndarray) -> np.ndarray:
        resized = cv2.resize(bgr_img, IMAGE_SIZE)
        normed = resized / 255.0
        return np.reshape(normed, (1, IMAGE_SIZE[0], IMAGE_SIZE[1], 3))

    def analyze(self, image_bytes: bytes) -> dict:
        img = self._decode_image(image_bytes)
        img_input = self._prepare_input(img)

        with self._model_lock:
            pred = float(self.model.predict(img_input, verbose=0)[0][0])

        has_finding = pred > FINDING_THRESHOLD
        confidence = int(pred * 100) if has_finding else int((1 - pred) * 100)
        result_label = "ANOMALİ TESPİT EDİLDİ" if has_finding else "NORMAL"

        return {
            "result": result_label,
            "confidence": confidence,
            "finding": has_finding,
            "finding_type": None,        # ileri versiyonda doldurulacak
            "finding_type_confidence": None,
        }

    def _get_heatmap(self, img_input: np.ndarray) -> Optional[np.ndarray]:
        if not self.last_conv_layer:
            return None
        try:
            with self._model_lock:
                grad_model = Model(
                    inputs=self.model.inputs,
                    outputs=[
                        self.model.get_layer(self.last_conv_layer).output,
                        self.model.output,
                    ],
                )
                with tf.GradientTape() as tape:
                    conv_outputs, preds = grad_model(img_input)
                    pred_index = tf.argmax(preds[0])
                    class_channel = preds[:, pred_index]

                grads = tape.gradient(class_channel, conv_outputs)
                if grads is None:
                    return None

                pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
                heatmap = conv_outputs[0] @ pooled_grads[..., tf.newaxis]
                heatmap = tf.squeeze(heatmap)
                heatmap = tf.maximum(heatmap, 0)
                max_val = tf.math.reduce_max(heatmap)
                if max_val == 0:
                    return None
                return (heatmap / max_val).numpy()
        except Exception:
            return None

    def train(self, image_bytes: bytes, label: int, **kwargs) -> dict:
        # İleri versiyonda organ bazlı training eklenecek
        return {"success": False, "message": "Bu organ için training henüz desteklenmiyor."}