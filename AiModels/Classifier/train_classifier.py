import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

SEED = 42
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS_HEAD = 20
EPOCHS_FINE = 10

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"
MODEL_OUT = BASE_DIR / "classifier.keras"
INDICES_OUT = BASE_DIR / "class_indices.json"
PRIORS_OUT = BASE_DIR / "class_priors.json"


def _assert_dataset() -> None:
    if not DATASET_DIR.exists() or not DATASET_DIR.is_dir():
        raise FileNotFoundError(f"Dataset klasoru bulunamadi: {DATASET_DIR}")
    class_dirs = [p for p in DATASET_DIR.iterdir() if p.is_dir()]
    if len(class_dirs) < 2:
        raise ValueError("Dataset icinde en az 2 sinif klasoru olmali.")


def _compute_class_weights(labels: np.ndarray, num_classes: int) -> dict[int, float]:
    counts = np.bincount(labels, minlength=num_classes)
    total = counts.sum()
    weights = {}
    for idx, count in enumerate(counts):
        if count == 0:
            continue
        weights[idx] = float(total / (num_classes * count))
    return weights


def _compute_class_priors(labels: np.ndarray, num_classes: int) -> dict[int, float]:
    counts = np.bincount(labels, minlength=num_classes)
    total = counts.sum()
    if total == 0:
        return {}
    return {idx: float(count / total) for idx, count in enumerate(counts) if count > 0}


def main() -> None:
    _assert_dataset()

    tf.keras.utils.set_random_seed(SEED)

    train_gen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=15,
        zoom_range=0.1,
        horizontal_flip=True,
        validation_split=0.2,
    )
    val_gen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        validation_split=0.2,
    )

    train_data = train_gen.flow_from_directory(
        str(DATASET_DIR),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="training",
        shuffle=True,
        seed=SEED,
    )

    val_data = val_gen.flow_from_directory(
        str(DATASET_DIR),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="validation",
        shuffle=False,
        seed=SEED,
    )

    print(f"Siniflar: {train_data.class_indices}")
    print(f"Egitim: {train_data.samples} goruntu")
    print(f"Validasyon: {val_data.samples} goruntu")

    class_weights = _compute_class_weights(train_data.classes, train_data.num_classes)
    class_priors = _compute_class_priors(train_data.classes, train_data.num_classes)

    base = EfficientNetB0(input_shape=(224, 224, 3), include_top=False, weights="imagenet")
    base.trainable = False

    x = GlobalAveragePooling2D()(base.output)
    x = Dropout(0.3)(x)
    x = Dense(128, activation="relu")(x)
    output = Dense(train_data.num_classes, activation="softmax")(x)

    model = Model(inputs=base.input, outputs=output)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True, monitor="val_loss"),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=2, min_lr=1e-6),
        tf.keras.callbacks.ModelCheckpoint(
            str(MODEL_OUT),
            save_best_only=True,
            monitor="val_accuracy",
            mode="max",
        ),
    ]

    history_head = model.fit(
        train_data,
        validation_data=val_data,
        epochs=EPOCHS_HEAD,
        callbacks=callbacks,
        class_weight=class_weights,
    )

    base.trainable = True
    for layer in base.layers[:-20]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.0001),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    print("\nFine-tuning basliyor...")
    history_fine = model.fit(
        train_data,
        validation_data=val_data,
        epochs=EPOCHS_FINE,
        callbacks=callbacks,
        class_weight=class_weights,
    )

    best_head = max(history_head.history.get("val_accuracy", [0.0]))
    best_fine = max(history_fine.history.get("val_accuracy", [0.0]))
    best_val = max(best_head, best_fine)

    with open(INDICES_OUT, "w", encoding="utf-8") as f:
        json.dump(train_data.class_indices, f, ensure_ascii=False, indent=2)
    idx_to_label = {v: k for k, v in train_data.class_indices.items()}
    priors_by_label = {idx_to_label[idx]: prior for idx, prior in class_priors.items() if idx in idx_to_label}
    with open(PRIORS_OUT, "w", encoding="utf-8") as f:
        json.dump(priors_by_label, f, ensure_ascii=False, indent=2)

    print("\nEgitim tamamlandi.")
    print(f"En iyi val_accuracy: {best_val:.2%}")
    print(f"Model kayit: {MODEL_OUT}")
    print(f"Sinif index dosyasi: {INDICES_OUT}")
    print(f"Class prior dosyasi: {PRIORS_OUT}")


if __name__ == "__main__":
    main()

