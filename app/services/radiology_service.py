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

# Keep TensorFlow logs quieter in API mode.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam

BASE_DIR = Path(__file__).resolve().parents[2]
AI_MODELS_DIR = BASE_DIR / "AiModels"
MODEL_PATH_KERAS = AI_MODELS_DIR / "beyin_bt_modeli.keras"
MODEL_PATH_H5 = AI_MODELS_DIR / "beyin_bt_modeli.h5"
MODEL_BACKUP_DIR = AI_MODELS_DIR / "backups"
FEEDBACK_DATASET_DIR = AI_MODELS_DIR / "feedback_dataset"
TRAINING_LOG_PATH = AI_MODELS_DIR / "training_logs.jsonl"

IMAGE_SIZE = (224, 224)
BLEEDING_THRESHOLD = 0.5
MAX_REPLAY_SAMPLES_PER_CLASS = 24
EPOCHS_PER_REVIEW = 2
MAX_TYPE_SAMPLES = 600
TYPE_TOP_K = 7
TYPE_INDEX_TTL_SEC = 30
TYPE_CONFIDENCE_MIN = 40


class RadiologyService:
    def __init__(self):
        MODEL_BACKUP_DIR.mkdir(parents=True, exist_ok=True)
        FEEDBACK_DATASET_DIR.mkdir(parents=True, exist_ok=True)

        self._model_lock = threading.RLock()
        self._current_model_path: Path = MODEL_PATH_KERAS
        self.model = self._load_model()
        self.last_conv_layer = self._find_last_conv_layer()
        self.feature_model = self._build_feature_model()
        self._compile_model()
        self._type_index: list[dict[str, Any]] = []
        self._type_index_last_refresh = 0.0
        self._refresh_type_index(force=True)

    def _load_model(self):
        model_path = None
        if MODEL_PATH_KERAS.exists():
            model_path = MODEL_PATH_KERAS
        elif MODEL_PATH_H5.exists():
            model_path = MODEL_PATH_H5

        if model_path is None:
            raise FileNotFoundError(f"Model file not found: {MODEL_PATH_KERAS} or {MODEL_PATH_H5}")

        self._current_model_path = model_path
        return load_model(str(model_path), compile=False)

    def _compile_model(self):
        self.model.compile(
            optimizer=Adam(learning_rate=0.00001),
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )

    def _find_last_conv_layer(self) -> Optional[str]:
        for layer in reversed(self.model.layers):
            if "conv" in layer.name.lower():
                return layer.name
        return None

    def _build_feature_model(self):
        # Use penultimate layer as embedding source when available.
        try:
            if len(self.model.layers) >= 2:
                return Model(inputs=self.model.input, outputs=self.model.layers[-2].output)
            return Model(inputs=self.model.input, outputs=self.model.output)
        except Exception:
            return None

    def _decode_image(self, image_bytes: bytes) -> np.ndarray:
        np_arr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Image decode failed.")
        return img

    def _prepare_model_input(self, bgr_img: np.ndarray) -> np.ndarray:
        img_resized = cv2.resize(bgr_img, IMAGE_SIZE)
        img_norm = img_resized / 255.0
        return np.reshape(img_norm, (1, IMAGE_SIZE[0], IMAGE_SIZE[1], 3))

    def analyze(self, image_bytes: bytes) -> dict:
        img = self._decode_image(image_bytes)
        img_input = self._prepare_model_input(img)

        with self._model_lock:
            pred = float(self.model.predict(img_input, verbose=0)[0][0])

        is_bleeding = pred > BLEEDING_THRESHOLD
        confidence = int(pred * 100) if is_bleeding else int((1 - pred) * 100)
        bleeding_type = None
        bleeding_type_confidence = None
        if is_bleeding:
            bleeding_type, bleeding_type_confidence = self._predict_bleeding_type(img)

        result_label = "KANAMA" if is_bleeding else "NORMAL"
        if is_bleeding and bleeding_type:
            result_label = f"KANAMA ({bleeding_type})"

        return {
            "result": result_label,
            "confidence": confidence,
            "is_bleeding": bool(is_bleeding),
            "bleeding_type": bleeding_type,
            "bleeding_type_confidence": bleeding_type_confidence,
        }

    def _get_heatmap(self, img_input):
        if not self.last_conv_layer:
            return None

        try:
            with self._model_lock:
                grad_model = Model(
                    inputs=self.model.inputs,
                    outputs=[self.model.get_layer(self.last_conv_layer).output, self.model.output],
                )

                with tf.GradientTape() as tape:
                    raw_output = grad_model(img_input)
                    if not (isinstance(raw_output, (list, tuple)) and len(raw_output) == 2):
                        return None
                    conv_outputs, preds = raw_output

                    while isinstance(conv_outputs, (list, tuple)):
                        conv_outputs = conv_outputs[0]
                    while isinstance(preds, (list, tuple)):
                        preds = preds[0]

                    pred_index = tf.argmax(preds[0])
                    class_channel = preds[:, pred_index]

                grads = tape.gradient(class_channel, conv_outputs)
                if grads is None:
                    return None

                pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
                conv_out = conv_outputs[0]
                heatmap = conv_out @ pooled_grads[..., tf.newaxis]
                heatmap = tf.squeeze(heatmap)
                heatmap = tf.maximum(heatmap, 0)
                max_val = tf.math.reduce_max(heatmap)
                if max_val == 0:
                    return None
                return (heatmap / max_val).numpy()
        except Exception:
            return None

    def _safe_timestamp(self) -> str:
        return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")

    def _extract_embedding(self, bgr_img: np.ndarray) -> Optional[np.ndarray]:
        model_input = self._prepare_model_input(bgr_img)
        try:
            with self._model_lock:
                if self.feature_model is not None:
                    raw_vec = self.feature_model.predict(model_input, verbose=0)
                else:
                    raw_vec = model_input
            vec = np.asarray(raw_vec, dtype=np.float32).reshape(-1)
            norm = np.linalg.norm(vec)
            if norm <= 0:
                return None
            return vec / norm
        except Exception:
            return None

    def _refresh_type_index(self, force: bool = False):
        now = datetime.now(timezone.utc).timestamp()
        if not force and (now - self._type_index_last_refresh) < TYPE_INDEX_TTL_SEC:
            return

        bleeding_dir = FEEDBACK_DATASET_DIR / "bleeding"
        if not bleeding_dir.exists():
            self._type_index = []
            self._type_index_last_refresh = now
            return

        json_files = sorted(
            bleeding_dir.glob("*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )[:MAX_TYPE_SAMPLES]

        new_index: list[dict[str, Any]] = []
        for meta_path in json_files:
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                bleeding_type = meta.get("bleeding_type")
                if not bleeding_type:
                    continue

                image_path_str = meta.get("image_path")
                if image_path_str:
                    image_path = Path(image_path_str)
                else:
                    image_path = meta_path.with_suffix(".jpg")

                if not image_path.exists():
                    continue

                img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
                if img is None:
                    continue

                emb = self._extract_embedding(img)
                if emb is None:
                    continue

                new_index.append({"type": bleeding_type, "embedding": emb})
            except Exception:
                continue

        self._type_index = new_index
        self._type_index_last_refresh = now

    def _predict_bleeding_type(self, bgr_img: np.ndarray) -> tuple[Optional[str], Optional[int]]:
        self._refresh_type_index(force=False)
        if not self._type_index:
            return None, None

        query = self._extract_embedding(bgr_img)
        if query is None:
            return None, None

        scored: list[tuple[float, str]] = []
        for item in self._type_index:
            sim = float(np.dot(query, item["embedding"]))
            scored.append((sim, item["type"]))
        if not scored:
            return None, None

        scored.sort(key=lambda x: x[0], reverse=True)
        top_k = scored[:TYPE_TOP_K]

        weighted_votes: dict[str, float] = {}
        for sim, btype in top_k:
            weight = max(sim, 0.0)
            weighted_votes[btype] = weighted_votes.get(btype, 0.0) + weight

        if not weighted_votes:
            best_type = top_k[0][1]
            best_conf = int(max(0.0, min(1.0, top_k[0][0])) * 100)
            return best_type, best_conf

        best_type, best_score = max(weighted_votes.items(), key=lambda kv: kv[1])
        total_score = sum(weighted_votes.values())
        if total_score <= 0:
            return None, None

        confidence = int((best_score / total_score) * 100)
        if confidence < TYPE_CONFIDENCE_MIN:
            return None, confidence
        return best_type, confidence

    def _save_feedback_sample(
        self,
        image_bytes: bytes,
        label: int,
        bleeding_type: Optional[str],
        analysis_id: Optional[int],
        doctor_id: Optional[int],
    ) -> Optional[Path]:
        class_name = "bleeding" if label == 1 else "normal"
        class_dir = FEEDBACK_DATASET_DIR / class_name
        class_dir.mkdir(parents=True, exist_ok=True)

        ts = self._safe_timestamp()
        sample_name = f"{ts}_analysis_{analysis_id or 'na'}_doctor_{doctor_id or 'na'}"
        image_path = class_dir / f"{sample_name}.jpg"
        meta_path = class_dir / f"{sample_name}.json"

        try:
            img = self._decode_image(image_bytes)
            ok = cv2.imwrite(str(image_path), img)
            if not ok:
                return None

            meta = {
                "created_at_utc": ts,
                "analysis_id": analysis_id,
                "doctor_id": doctor_id,
                "label": label,
                "bleeding_type": bleeding_type,
                "image_path": str(image_path),
            }
            meta_path.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
            return image_path
        except Exception:
            return None

    def _read_recent_class_samples(self, class_name: str, label: int, max_samples: int) -> list[tuple[np.ndarray, int]]:
        class_dir = FEEDBACK_DATASET_DIR / class_name
        if not class_dir.exists():
            return []

        image_files = sorted(class_dir.glob("*.jpg"), key=lambda p: p.stat().st_mtime, reverse=True)
        samples: list[tuple[np.ndarray, int]] = []
        for img_path in image_files[:max_samples]:
            img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if img is None:
                continue
            samples.append((img, label))
        return samples

    def _augment(self, bgr_img: np.ndarray) -> list[np.ndarray]:
        alpha = random.uniform(0.9, 1.1)
        beta = random.randint(-12, 12)
        bright = cv2.convertScaleAbs(bgr_img, alpha=alpha, beta=beta)
        return [
            bgr_img,
            cv2.flip(bgr_img, 1),
            cv2.rotate(bgr_img, cv2.ROTATE_90_CLOCKWISE),
            bright,
        ]

    def _build_training_batch(self) -> tuple[np.ndarray, np.ndarray]:
        bleeding = self._read_recent_class_samples("bleeding", 1, MAX_REPLAY_SAMPLES_PER_CLASS)
        normal = self._read_recent_class_samples("normal", 0, MAX_REPLAY_SAMPLES_PER_CLASS)
        samples = bleeding + normal
        random.shuffle(samples)
        if not samples:
            raise ValueError("No feedback samples available for training.")

        x_items: list[np.ndarray] = []
        y_items: list[list[float]] = []
        for img, label in samples:
            augmented = self._augment(img)
            for aug in augmented:
                resized = cv2.resize(aug, IMAGE_SIZE) / 255.0
                x_items.append(resized.astype(np.float32))
                y_items.append([float(label)])

        x_train = np.asarray(x_items, dtype=np.float32)
        y_train = np.asarray(y_items, dtype=np.float32)
        return x_train, y_train

    def _backup_model(self) -> tuple[Path, Path]:
        source_path = self._current_model_path
        suffix = source_path.suffix or ".keras"
        backup_name = f"beyin_bt_modeli_{self._safe_timestamp()}{suffix}"
        backup_path = MODEL_BACKUP_DIR / backup_name
        shutil.copy2(source_path, backup_path)
        return backup_path, source_path

    def _atomic_save_model(self):
        tmp_path = AI_MODELS_DIR / "beyin_bt_modeli.tmp.keras"
        self.model.save(str(tmp_path))
        os.replace(tmp_path, MODEL_PATH_KERAS)
        self._current_model_path = MODEL_PATH_KERAS

    def _reload_model(self):
        self.model = self._load_model()
        self.last_conv_layer = self._find_last_conv_layer()
        self.feature_model = self._build_feature_model()
        self._compile_model()
        self._refresh_type_index(force=True)

    def _log_training_event(self, payload: dict[str, Any]):
        payload.setdefault("timestamp_utc", datetime.now(timezone.utc).isoformat())
        try:
            with TRAINING_LOG_PATH.open("a", encoding="utf-8") as fp:
                fp.write(json.dumps(payload, ensure_ascii=False) + "\n")
        except Exception:
            # Logging should never block training flow.
            pass

    def train(
        self,
        image_bytes: bytes,
        label: int,
        bleeding_type: Optional[str] = None,
        analysis_id: Optional[int] = None,
        doctor_id: Optional[int] = None,
    ) -> dict:
        if label not in (0, 1):
            return {"success": False, "message": "Label must be 0 or 1."}

        sample_path = self._save_feedback_sample(
            image_bytes=image_bytes,
            label=label,
            bleeding_type=bleeding_type,
            analysis_id=analysis_id,
            doctor_id=doctor_id,
        )

        if sample_path is None:
            return {"success": False, "message": "Feedback sample could not be saved."}

        with self._model_lock:
            backup_path: Optional[Path] = None
            backup_source_path: Optional[Path] = None
            try:
                x_train, y_train = self._build_training_batch()
                backup_path, backup_source_path = self._backup_model()

                self.model.fit(
                    x_train,
                    y_train,
                    epochs=EPOCHS_PER_REVIEW,
                    batch_size=min(16, len(x_train)),
                    verbose=0,
                    shuffle=True,
                )
                self._atomic_save_model()
                self._reload_model()

                event = {
                    "success": True,
                    "analysis_id": analysis_id,
                    "doctor_id": doctor_id,
                    "label": label,
                    "bleeding_type": bleeding_type,
                    "sample_path": str(sample_path),
                    "samples_used": int(len(x_train)),
                    "type_index_size": len(self._type_index),
                    "backup_path": str(backup_path) if backup_path else None,
                }
                self._log_training_event(event)
                return {"success": True, "message": "Model training completed.", **event}
            except Exception as exc:
                # Fail-safe rollback to last backup.
                if backup_path and backup_source_path and backup_path.exists():
                    shutil.copy2(backup_path, backup_source_path)
                    self._current_model_path = backup_source_path
                    try:
                        self._reload_model()
                    except Exception:
                        pass

                event = {
                    "success": False,
                    "analysis_id": analysis_id,
                    "doctor_id": doctor_id,
                    "label": label,
                    "bleeding_type": bleeding_type,
                    "sample_path": str(sample_path),
                    "type_index_size": len(self._type_index),
                    "backup_path": str(backup_path) if backup_path else None,
                    "error": str(exc),
                }
                self._log_training_event(event)
                return {"success": False, "message": "Model training failed.", **event}
