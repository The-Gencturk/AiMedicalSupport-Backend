import random
import shutil
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam

SEED = 42
IMAGE_SIZE = (224, 224)
MAX_SAMPLES_PER_CLASS = 400
MIN_SAMPLES_PER_CLASS = 20
VAL_RATIO = 0.2
EPOCHS = 10
BATCH_SIZE = 16

BASE_DIR = Path(__file__).resolve().parent
MODEL_KERAS = BASE_DIR / "beyin_bt_modeli.keras"
MODEL_H5 = BASE_DIR / "beyin_bt_modeli.h5"
DATASET_DIR = BASE_DIR / "feedback_dataset"
BACKUP_DIR = BASE_DIR / "backups"


def _safe_ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def _active_model_path() -> Path:
    if MODEL_KERAS.exists():
        return MODEL_KERAS
    if MODEL_H5.exists():
        return MODEL_H5
    raise FileNotFoundError(f"Model bulunamadi: {MODEL_KERAS} veya {MODEL_H5}")


def _augment(img: np.ndarray) -> list[np.ndarray]:
    bright = cv2.convertScaleAbs(img, alpha=random.uniform(0.9, 1.1), beta=random.randint(-10, 10))
    return [img, cv2.flip(img, 1), bright]


def _read_class_samples(class_name: str, label: int, limit: int) -> list[tuple[np.ndarray, int]]:
    class_dir = DATASET_DIR / class_name
    if not class_dir.exists():
        return []

    files = sorted(class_dir.glob("*.jpg"), key=lambda p: p.stat().st_mtime, reverse=True)[:limit]
    samples = []
    for file in files:
        img = cv2.imread(str(file), cv2.IMREAD_COLOR)
        if img is not None:
            samples.append((img, label))
    return samples


def _build_dataset() -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    bleeding = _read_class_samples("bleeding", 1, MAX_SAMPLES_PER_CLASS)
    normal = _read_class_samples("normal", 0, MAX_SAMPLES_PER_CLASS)

    if len(bleeding) < MIN_SAMPLES_PER_CLASS or len(normal) < MIN_SAMPLES_PER_CLASS:
        raise ValueError(
            f"Yetersiz veri. bleeding={len(bleeding)}, normal={len(normal)}, minimum={MIN_SAMPLES_PER_CLASS}"
        )

    items = bleeding + normal
    random.shuffle(items)

    x_items = []
    y_items = []
    for img, label in items:
        for aug in _augment(img):
            norm = (cv2.resize(aug, IMAGE_SIZE) / 255.0).astype(np.float32)
            x_items.append(norm)
            y_items.append([float(label)])

    x = np.asarray(x_items)
    y = np.asarray(y_items)
    return x, y, {"bleeding": len(bleeding), "normal": len(normal)}


def _split_train_val(x: np.ndarray, y: np.ndarray):
    total = len(x)
    val_size = max(8, int(total * VAL_RATIO))
    train_size = total - val_size
    if train_size < 8:
        raise ValueError("Train/val ayrimi icin veri yetersiz.")

    indices = np.random.permutation(total)
    val_idx = indices[:val_size]
    train_idx = indices[val_size:]
    return x[train_idx], y[train_idx], x[val_idx], y[val_idx]


def main() -> None:
    tf.keras.utils.set_random_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)

    model_path = _active_model_path()
    model = load_model(str(model_path), compile=False)
    model.compile(optimizer=Adam(learning_rate=1e-5), loss="binary_crossentropy", metrics=["accuracy"])

    x_all, y_all, counts = _build_dataset()
    x_train, y_train, x_val, y_val = _split_train_val(x_all, y_all)

    before = model.evaluate(x_val, y_val, verbose=0, return_dict=True)

    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    backup_path = BACKUP_DIR / f"beyin_bt_modeli_{_safe_ts()}{model_path.suffix}"
    shutil.copy2(model_path, backup_path)

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=1, min_lr=1e-6),
        ModelCheckpoint(str(MODEL_KERAS), monitor="val_accuracy", mode="max", save_best_only=True),
    ]

    model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=EPOCHS,
        batch_size=min(BATCH_SIZE, len(x_train)),
        verbose=1,
        shuffle=True,
        callbacks=callbacks,
    )

    reloaded = load_model(str(MODEL_KERAS), compile=False)
    reloaded.compile(optimizer=Adam(learning_rate=1e-5), loss="binary_crossentropy", metrics=["accuracy"])
    after = reloaded.evaluate(x_val, y_val, verbose=0, return_dict=True)

    print("Sinif sayilari:", counts)
    print("Val once:", {k: float(v) for k, v in before.items()})
    print("Val sonra:", {k: float(v) for k, v in after.items()})
    print("Model kaydedildi:", MODEL_KERAS)
    print("Yedek alindi:", backup_path)


if __name__ == "__main__":
    main()

