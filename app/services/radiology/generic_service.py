# app/services/radiology/generic_service.py
import json
import os
import threading
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam

from app.services.radiology.base_radiology import BaseRadiologyService

IMAGE_SIZE = (224, 224)
FINDING_THRESHOLD = 0.5
LOW_CONFIDENCE_THRESHOLD = 60
BORDERLINE_MARGIN = 0.08


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

        # Load optional class index mapping from model directory.
        model_dir = Path(model_path).parent
        indices_path = model_dir / f"{organ_name}_class_indices.json"
        self._class_indices: Optional[dict[int, str]] = None
        self._finding_index: Optional[int] = None

        if indices_path.exists():
            with open(indices_path, encoding="utf-8") as f:
                raw = json.load(f)

            if raw and isinstance(next(iter(raw)), str) and not next(iter(raw)).lstrip("-").isdigit():
                # {"cancer": 0, "normal": 1} -> {0: "cancer", 1: "normal"}
                self._class_indices = {int(v): str(k) for k, v in raw.items()}
            else:
                # {"0": "cancer", "1": "normal"}
                self._class_indices = {int(k): str(v) for k, v in raw.items()}

            for idx, label in self._class_indices.items():
                if "normal" not in label.lower():
                    self._finding_index = idx
                    break

        self._warmup()

    def _warmup(self):
        """Run a dummy prediction at startup to warm the graph."""
        try:
            dummy = np.zeros((1, IMAGE_SIZE[0], IMAGE_SIZE[1], 3), dtype=np.float32)
            with self._model_lock:
                self.model.predict(dummy, verbose=0)
        except Exception:
            pass

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
        normed = (resized / 255.0).astype(np.float32)
        return np.reshape(normed, (1, IMAGE_SIZE[0], IMAGE_SIZE[1], 3))

    def analyze(self, image_bytes: bytes) -> dict:
        img = self._decode_image(image_bytes)
        img_input = self._prepare_input(img)

        with self._model_lock:
            raw_pred = self.model.predict(img_input, verbose=0)[0]

        is_binary = raw_pred.shape[-1] == 1
        pred_scalar = float(raw_pred[0]) if is_binary else None

        has_finding = False
        confidence = 0
        finding_type = None
        needs_review = False
        review_reason = None

        if self._class_indices:
            if is_binary:
                # If finding class is index 0, low sigmoid values mean finding.
                if self._finding_index == 0:
                    has_finding = pred_scalar < FINDING_THRESHOLD
                    confidence = int((1 - pred_scalar) * 100) if has_finding else int(pred_scalar * 100)
                else:
                    has_finding = pred_scalar >= FINDING_THRESHOLD
                    confidence = int(pred_scalar * 100) if has_finding else int((1 - pred_scalar) * 100)

                needs_review = (
                    pred_scalar is not None
                    and (
                        abs(pred_scalar - FINDING_THRESHOLD) < BORDERLINE_MARGIN
                        or confidence < LOW_CONFIDENCE_THRESHOLD
                    )
                )

                if has_finding:
                    finding_label = self._class_indices.get(self._finding_index)
                    finding_type = finding_label.upper() if finding_label else None
            else:
                predicted_idx = int(np.argmax(raw_pred))
                predicted_label = self._class_indices.get(predicted_idx, "unknown")
                has_finding = "normal" not in predicted_label.lower()
                confidence = int(float(raw_pred[predicted_idx]) * 100)
                needs_review = confidence < LOW_CONFIDENCE_THRESHOLD
                if has_finding and predicted_label != "unknown":
                    finding_type = predicted_label.upper()

            result_label = finding_type if has_finding and finding_type else "NORMAL"
        else:
            has_finding = (pred_scalar or 0.0) > FINDING_THRESHOLD
            confidence = int((pred_scalar or 0.0) * 100) if has_finding else int((1 - (pred_scalar or 0.0)) * 100)
            if pred_scalar is not None:
                needs_review = (
                    abs(pred_scalar - FINDING_THRESHOLD) < BORDERLINE_MARGIN
                    or confidence < LOW_CONFIDENCE_THRESHOLD
                )
            result_label = "ANOMALI TESPIT EDILDI" if has_finding else "NORMAL"

        if needs_review:
            if pred_scalar is not None and abs(pred_scalar - FINDING_THRESHOLD) < BORDERLINE_MARGIN:
                review_reason = "borderline_probability"
            else:
                review_reason = "low_confidence"

        return {
            "result": result_label,
            "confidence": confidence,
            "finding": has_finding,
            "finding_type": finding_type,
            "is_bleeding": False,
            "bleeding_type": None,
            "needs_review": needs_review,
            "review_reason": review_reason,
        }

    def _get_heatmap(self, img_input: np.ndarray) -> Optional[np.ndarray]:
        """
        Grad-CAM support for both binary(sigmoid) and multi-class models.
        """
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
                img_tensor = tf.cast(img_input, tf.float32)

                with tf.GradientTape() as tape:
                    tape.watch(img_tensor)
                    conv_outputs, preds = grad_model(img_tensor)

                    output_shape = preds.shape
                    if len(output_shape) == 2 and output_shape[-1] == 1:
                        class_channel = preds[:, 0]
                    else:
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
        except Exception as e:
            print(f"[{self.organ_name}] Heatmap HATA: {e}")
            return None

    def train(self, image_bytes: bytes, label: int, **kwargs) -> dict:
        return {
            "success": False,
            "message": f"'{self.organ_name}' organi icin egitim henuz desteklenmiyor.",
            "organ_name": self.organ_name,
            "attempted": False,
        }

