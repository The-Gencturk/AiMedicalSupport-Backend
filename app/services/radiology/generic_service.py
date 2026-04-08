# app/services/radiology/generic_service.py
import json
import os
import random
import shutil
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

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
MAX_REPLAY_SAMPLES_PER_CLASS = 24
EPOCHS_PER_REVIEW = 2
MIN_CLASS_SAMPLES_FOR_TRAIN = 1
VALIDATION_SPLIT = 0.2
MIN_VAL_SAMPLES = 8
MAX_ALLOWED_VAL_LOSS_INCREASE = 0.20
MAX_ALLOWED_VAL_ACC_DROP = 0.05


class GenericRadiologyService(BaseRadiologyService):

    def __init__(self, model_path: str, organ_name: str):
        self.organ_name = organ_name
        self._model_lock = threading.RLock()
        self._current_model_path = Path(model_path).resolve()
        self._model_dir = self._current_model_path.parent
        self._feedback_dataset_dir = self._model_dir / "feedback_dataset"
        self._model_backup_dir = self._model_dir / "backups"
        self._training_log_path = self._model_dir / "training_logs.jsonl"

        self._feedback_dataset_dir.mkdir(parents=True, exist_ok=True)
        self._model_backup_dir.mkdir(parents=True, exist_ok=True)

        self.model = load_model(str(self._current_model_path), compile=False)
        self._input_h, self._input_w, self._input_c = self._resolve_model_input_shape()
        self._compile_model()
        self.last_conv_layer = self._find_last_conv_layer()

        # Load optional class index mapping from model directory.
        indices_path = self._model_dir / f"{organ_name}_class_indices.json"
        self._class_indices: Optional[dict[int, str]] = None
        self._finding_index: Optional[int] = None
        self._normal_index: Optional[int] = None

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
                if "normal" in label.lower() and self._normal_index is None:
                    self._normal_index = idx
                if "normal" not in label.lower() and self._finding_index is None:
                    self._finding_index = idx

        if self._normal_index is None and self._class_indices:
            self._normal_index = min(self._class_indices.keys())
        if self._finding_index is None and self._class_indices:
            non_normal = [idx for idx, lbl in self._class_indices.items() if "normal" not in lbl.lower()]
            self._finding_index = non_normal[0] if non_normal else max(self._class_indices.keys())

        self._warmup()

    def _resolve_model_input_shape(self) -> tuple[int, int, int]:
        shape = getattr(self.model, "input_shape", None)
        if isinstance(shape, list) and shape:
            shape = shape[0]
        if not shape or len(shape) < 4:
            return IMAGE_SIZE[0], IMAGE_SIZE[1], 3
        h = int(shape[1]) if shape[1] else IMAGE_SIZE[0]
        w = int(shape[2]) if shape[2] else IMAGE_SIZE[1]
        c = int(shape[3]) if shape[3] else 3
        return h, w, c

    def _compile_model(self):
        output_shape = self.model.output_shape
        if isinstance(output_shape, list):
            output_shape = output_shape[0]
        output_dim = int(output_shape[-1]) if output_shape and len(output_shape) >= 2 and output_shape[-1] else 1
        loss = "binary_crossentropy" if output_dim == 1 else "categorical_crossentropy"
        self.model.compile(
            optimizer=Adam(learning_rate=0.00001),
            loss=loss,
            metrics=["accuracy"],
        )

    def _warmup(self):
        """Run a dummy prediction at startup to warm the graph."""
        try:
            dummy = np.zeros((1, self._input_h, self._input_w, self._input_c), dtype=np.float32)
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
        prepared = self._prepare_image_tensor(bgr_img)
        return np.reshape(prepared, (1, self._input_h, self._input_w, self._input_c))

    def _prepare_image_tensor(self, bgr_img: np.ndarray) -> np.ndarray:
        resized = cv2.resize(bgr_img, (self._input_w, self._input_h))
        if self._input_c == 1:
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            arr = (gray / 255.0).astype(np.float32)
            return np.reshape(arr, (self._input_h, self._input_w, 1))

        arr = (resized / 255.0).astype(np.float32)
        if self._input_c > 3:
            extras = np.repeat(arr[:, :, :1], self._input_c - 3, axis=2)
            arr = np.concatenate([arr, extras], axis=2)
        elif self._input_c < 3:
            arr = arr[:, :, :self._input_c]
        return arr

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
            return self._get_input_saliency_heatmap(img_input)
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
                        positive = preds[:, 0]
                        class_channel = tf.where(positive >= 0.5, positive, 1.0 - positive)
                    else:
                        pred_index = tf.argmax(preds[0])
                        class_channel = preds[:, pred_index]

                grads = tape.gradient(class_channel, conv_outputs)
                if grads is None:
                    return self._get_input_saliency_heatmap(img_input)

                pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
                heatmap = conv_outputs[0] @ pooled_grads[..., tf.newaxis]
                heatmap = tf.squeeze(heatmap)
                heatmap = tf.maximum(heatmap, 0)
                max_val = tf.math.reduce_max(heatmap)
                if max_val == 0:
                    return self._get_input_saliency_heatmap(img_input)
                return (heatmap / max_val).numpy()
        except Exception as e:
            print(f"[{self.organ_name}] Heatmap HATA: {e}")
            return self._get_input_saliency_heatmap(img_input)

    def _get_input_saliency_heatmap(self, img_input: np.ndarray) -> Optional[np.ndarray]:
        """
        Fallback heatmap: gradient saliency on the input image.
        """
        try:
            with self._model_lock:
                img_tensor = tf.cast(img_input, tf.float32)
                with tf.GradientTape() as tape:
                    tape.watch(img_tensor)
                    preds = self.model(img_tensor, training=False)
                    output_shape = preds.shape
                    if len(output_shape) == 2 and output_shape[-1] == 1:
                        positive = preds[:, 0]
                        class_channel = tf.where(positive >= 0.5, positive, 1.0 - positive)
                    else:
                        pred_index = tf.argmax(preds[0])
                        class_channel = preds[:, pred_index]

                grads = tape.gradient(class_channel, img_tensor)
                if grads is None:
                    return None

                saliency = tf.reduce_mean(tf.abs(grads[0]), axis=-1)
                saliency = tf.maximum(saliency, 0)
                max_val = tf.math.reduce_max(saliency)
                if max_val == 0:
                    return None
                return (saliency / max_val).numpy()
        except Exception:
            return None

    def _safe_timestamp(self) -> str:
        return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")

    def _save_feedback_sample(
        self,
        image_bytes: bytes,
        label: int,
        finding_type: Optional[str],
        analysis_id: Optional[int],
        doctor_id: Optional[int],
    ) -> Optional[Path]:
        class_name = "finding" if label == 1 else "normal"
        class_dir = self._feedback_dataset_dir / class_name
        class_dir.mkdir(parents=True, exist_ok=True)

        ts = self._safe_timestamp()
        sample_name = f"{ts}_analysis_{analysis_id or 'na'}_doctor_{doctor_id or 'na'}"
        image_path = class_dir / f"{sample_name}.jpg"
        meta_path = class_dir / f"{sample_name}.json"

        try:
            img = self._decode_image(image_bytes)
            if not cv2.imwrite(str(image_path), img):
                return None

            meta = {
                "created_at_utc": ts,
                "analysis_id": analysis_id,
                "doctor_id": doctor_id,
                "label": label,
                "finding_type": finding_type,
                "image_path": str(image_path),
            }
            meta_path.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
            return image_path
        except Exception:
            return None

    def _read_recent_class_samples(
        self,
        class_name: str,
        label: int,
        max_samples: int,
    ) -> list[tuple[np.ndarray, int, Optional[str]]]:
        class_dir = self._feedback_dataset_dir / class_name
        if not class_dir.exists():
            return []

        image_files = sorted(class_dir.glob("*.jpg"), key=lambda p: p.stat().st_mtime, reverse=True)
        samples: list[tuple[np.ndarray, int, Optional[str]]] = []
        for img_path in image_files[:max_samples]:
            img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if img is None:
                continue
            finding_type = None
            meta_path = img_path.with_suffix(".json")
            if meta_path.exists():
                try:
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
                    raw_type = meta.get("finding_type")
                    if isinstance(raw_type, str) and raw_type.strip():
                        finding_type = raw_type.strip()
                except Exception:
                    finding_type = None
            samples.append((img, label, finding_type))
        return samples

    def _augment(self, bgr_img: np.ndarray) -> list[np.ndarray]:
        bright = cv2.convertScaleAbs(bgr_img, alpha=random.uniform(0.9, 1.1), beta=random.randint(-12, 12))
        return [bgr_img, cv2.flip(bgr_img, 1), cv2.rotate(bgr_img, cv2.ROTATE_90_CLOCKWISE), bright]

    def _label_to_target(self, label: int, finding_type: Optional[str] = None) -> np.ndarray:
        output_shape = self.model.output_shape
        if isinstance(output_shape, list):
            output_shape = output_shape[0]
        output_dim = int(output_shape[-1]) if output_shape and len(output_shape) >= 2 and output_shape[-1] else 1

        if output_dim == 1:
            return np.asarray([float(label)], dtype=np.float32)

        target_idx = None
        if label == 0:
            target_idx = self._normal_index
        else:
            target_idx = self._finding_index
            if finding_type and self._class_indices:
                needle = finding_type.strip().lower()
                for idx, class_label in self._class_indices.items():
                    class_lower = class_label.lower()
                    if class_lower == needle or needle in class_lower:
                        target_idx = idx
                        break

        if target_idx is None:
            target_idx = 0 if label == 0 else min(1, output_dim - 1)

        target_idx = max(0, min(int(target_idx), output_dim - 1))
        vec = np.zeros((output_dim,), dtype=np.float32)
        vec[target_idx] = 1.0
        return vec

    def _build_training_batch(self) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
        finding_samples = self._read_recent_class_samples("finding", 1, MAX_REPLAY_SAMPLES_PER_CLASS)
        normal_samples = self._read_recent_class_samples("normal", 0, MAX_REPLAY_SAMPLES_PER_CLASS)

        samples = finding_samples + normal_samples
        random.shuffle(samples)
        if not samples:
            raise ValueError("No feedback samples available.")

        x_items: list[np.ndarray] = []
        y_items: list[np.ndarray] = []
        for img, label, finding_type in samples:
            target = self._label_to_target(label, finding_type=finding_type)
            for aug in self._augment(img):
                x_items.append(self._prepare_image_tensor(aug))
                y_items.append(target)

        class_counts = {
            "finding": len(finding_samples),
            "normal": len(normal_samples),
        }

        return np.asarray(x_items, dtype=np.float32), np.asarray(y_items, dtype=np.float32), class_counts

    def _split_train_val(self, x_all: np.ndarray, y_all: np.ndarray):
        total = len(x_all)
        if total < MIN_VAL_SAMPLES:
            return x_all, y_all, None, None

        val_size = max(4, int(total * VALIDATION_SPLIT))
        train_size = total - val_size
        if train_size < 4:
            return x_all, y_all, None, None

        indices = np.random.permutation(total)
        val_idx = indices[:val_size]
        train_idx = indices[val_size:]

        x_train = x_all[train_idx]
        y_train = y_all[train_idx]
        x_val = x_all[val_idx]
        y_val = y_all[val_idx]
        return x_train, y_train, x_val, y_val

    def _evaluate_on_batch(self, x_data: np.ndarray, y_data: np.ndarray) -> Optional[dict[str, float]]:
        if x_data is None or y_data is None or len(x_data) == 0:
            return None
        metrics = self.model.evaluate(x_data, y_data, verbose=0, return_dict=True)
        return {
            "loss": float(metrics.get("loss", 0.0)),
            "accuracy": float(metrics.get("accuracy", 0.0)),
        }

    def _is_quality_regression(
        self,
        before_metrics: Optional[dict[str, float]],
        after_metrics: Optional[dict[str, float]],
    ) -> bool:
        if not before_metrics or not after_metrics:
            return False

        before_loss = before_metrics["loss"]
        after_loss = after_metrics["loss"]
        before_acc = before_metrics["accuracy"]
        after_acc = after_metrics["accuracy"]

        loss_degraded = after_loss > (before_loss * (1 + MAX_ALLOWED_VAL_LOSS_INCREASE))
        acc_degraded = after_acc < (before_acc - MAX_ALLOWED_VAL_ACC_DROP)
        return loss_degraded and acc_degraded

    def _backup_model(self) -> tuple[Path, Path]:
        source_path = self._current_model_path
        suffix = source_path.suffix or ".keras"
        backup_name = f"{source_path.stem}_{self._safe_timestamp()}{suffix}"
        backup_path = self._model_backup_dir / backup_name
        shutil.copy2(source_path, backup_path)
        return backup_path, source_path

    def _atomic_save_model(self):
        target = self._current_model_path
        suffix = target.suffix if target.suffix else ".keras"
        tmp_path = target.with_name(f"{target.stem}.tmp{suffix}")
        self.model.save(str(tmp_path))
        os.replace(tmp_path, target)

    def _reload_model(self):
        self.model = load_model(str(self._current_model_path), compile=False)
        self._input_h, self._input_w, self._input_c = self._resolve_model_input_shape()
        self._compile_model()
        self.last_conv_layer = self._find_last_conv_layer()

    def _log_training_event(self, payload: dict[str, Any]):
        payload.setdefault("timestamp_utc", datetime.now(timezone.utc).isoformat())
        try:
            with self._training_log_path.open("a", encoding="utf-8") as fp:
                fp.write(json.dumps(payload, ensure_ascii=False) + "\n")
        except Exception:
            pass

    def train(self, image_bytes: bytes, label: int, **kwargs) -> dict:
        if label not in (0, 1):
            return {"success": False, "message": "Label must be 0 or 1.", "attempted": False}

        finding_type = kwargs.get("finding_type") or kwargs.get("bleeding_type")
        if label == 0:
            finding_type = None

        analysis_id = kwargs.get("analysis_id")
        doctor_id = kwargs.get("doctor_id")

        sample_path = self._save_feedback_sample(
            image_bytes=image_bytes,
            label=label,
            finding_type=finding_type,
            analysis_id=analysis_id,
            doctor_id=doctor_id,
        )
        if sample_path is None:
            return {
                "success": False,
                "attempted": True,
                "message": "Feedback sample could not be saved.",
                "organ_name": self.organ_name,
            }

        with self._model_lock:
            backup_path = None
            backup_source_path = None
            try:
                x_all, y_all, class_counts = self._build_training_batch()
                if (
                    class_counts["finding"] < MIN_CLASS_SAMPLES_FOR_TRAIN
                    or class_counts["normal"] < MIN_CLASS_SAMPLES_FOR_TRAIN
                ):
                    missing_counts = {
                        "finding": max(0, MIN_CLASS_SAMPLES_FOR_TRAIN - class_counts["finding"]),
                        "normal": max(0, MIN_CLASS_SAMPLES_FOR_TRAIN - class_counts["normal"]),
                    }
                    event = {
                        "success": False,
                        "attempted": True,
                        "organ_name": self.organ_name,
                        "analysis_id": analysis_id,
                        "doctor_id": doctor_id,
                        "label": label,
                        "finding_type": finding_type,
                        "message": "Not enough balanced feedback samples yet.",
                        "class_counts": class_counts,
                        "min_required_per_class": MIN_CLASS_SAMPLES_FOR_TRAIN,
                        "missing_counts": missing_counts,
                    }
                    self._log_training_event(event)
                    return event

                x_train, y_train, x_val, y_val = self._split_train_val(x_all, y_all)
                before_metrics = self._evaluate_on_batch(x_val, y_val)
                backup_path, backup_source_path = self._backup_model()

                fit_kwargs = {
                    "x": x_train,
                    "y": y_train,
                    "epochs": EPOCHS_PER_REVIEW,
                    "batch_size": min(16, len(x_train)),
                    "verbose": 0,
                    "shuffle": True,
                }
                if x_val is not None and y_val is not None:
                    fit_kwargs["validation_data"] = (x_val, y_val)

                self.model.fit(**fit_kwargs)

                after_metrics = self._evaluate_on_batch(x_val, y_val)
                if self._is_quality_regression(before_metrics, after_metrics):
                    raise RuntimeError("Quality gate failed: validation metrics regressed.")

                self._atomic_save_model()
                self._reload_model()

                event = {
                    "success": True,
                    "attempted": True,
                    "organ_name": self.organ_name,
                    "analysis_id": analysis_id,
                    "doctor_id": doctor_id,
                    "label": label,
                    "finding_type": finding_type,
                    "samples_used": int(len(x_train)),
                    "class_counts": class_counts,
                    "model_path": str(self._current_model_path),
                    "backup_path": str(backup_path) if backup_path else None,
                }
                if before_metrics and after_metrics:
                    event["val_before"] = before_metrics
                    event["val_after"] = after_metrics

                self._log_training_event(event)
                return {"message": "Model training completed.", **event}

            except Exception as exc:
                if backup_path and backup_source_path and backup_path.exists():
                    shutil.copy2(backup_path, backup_source_path)
                    self._current_model_path = backup_source_path
                    try:
                        self._reload_model()
                    except Exception:
                        pass

                event = {
                    "success": False,
                    "attempted": True,
                    "organ_name": self.organ_name,
                    "analysis_id": analysis_id,
                    "doctor_id": doctor_id,
                    "label": label,
                    "finding_type": finding_type,
                    "model_path": str(self._current_model_path),
                    "backup_path": str(backup_path) if backup_path else None,
                    "error": str(exc),
                }
                self._log_training_event(event)
                return {"message": "Model training failed.", **event}
