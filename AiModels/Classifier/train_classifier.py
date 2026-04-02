import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from pathlib import Path
import json

DATASET_DIR = "dataset"
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 20

train_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    zoom_range=0.1,
    horizontal_flip=True,
    validation_split=0.2
)

train_data = train_gen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training"
)

val_data = train_gen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation"
)

print(f"Sınıflar: {train_data.class_indices}")
print(f"Eğitim: {train_data.samples} görüntü")
print(f"Validasyon: {val_data.samples} görüntü")

base = EfficientNetB0(input_shape=(224,224,3), include_top=False, weights="imagenet")
base.trainable = False

x = GlobalAveragePooling2D()(base.output)
x = Dropout(0.3)(x)
x = Dense(128, activation="relu")(x)
output = Dense(train_data.num_classes, activation="softmax")(x)

model = Model(inputs=base.input, outputs=output)
model.compile(
    optimizer=tf.keras.optimizers.Adam(0.001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(
            "classifier.keras",
            save_best_only=True,
            monitor="val_accuracy"
        )
    ]
)
base.trainable = True
for layer in base.layers[:-20]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(0.0001),  # daha düşük lr
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

print("\nFine-tuning başlıyor...")
history2 = model.fit(
    train_data,
    validation_data=val_data,
    epochs=10,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(
            "classifier.keras",
            save_best_only=True,
            monitor="val_accuracy"
        )
    ]
)

print(f"Fine-tuning sonrası val_accuracy: {max(history2.history['val_accuracy']):.2%}")

with open("class_indices.json", "w") as f:
    json.dump(train_data.class_indices, f, ensure_ascii=False)

print(f"\nEğitim tamamlandı!")
print(f"En iyi val_accuracy: {max(history.history['val_accuracy']):.2%}")
print(f"Sınıf indexleri: {train_data.class_indices}")