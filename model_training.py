# model_training.py  (FINAL FIXED VERSION)

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# ==========================================================
# 1. CONFIGURATION
# ==========================================================

IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 10   # Increased for better learning

DATA_DIR = "data/Data"  # <<—— FINAL, CORRECT PATH

train_dir = os.path.join(DATA_DIR, "train")
val_dir = os.path.join(DATA_DIR, "valid")
test_dir = os.path.join(DATA_DIR, "test")

print("Training Folder:", train_dir)
print("Validation Folder:", val_dir)
print("Testing Folder:", test_dir)


# ==========================================================
# 2. DATA GENERATORS (IMPROVED)
# ==========================================================

# HEAVY AUGMENTATION FIXES OVERFITTING + CLASS IMBALANCE
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=20,
    width_shift_range=0.15,
    height_shift_range=0.15,
    zoom_range=0.15,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

val_data = val_datagen.flow_from_directory(
    val_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

test_data = test_datagen.flow_from_directory(
    test_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

NUM_CLASSES = train_data.num_classes
print("Number of Classes:", NUM_CLASSES)
print("Class Index Mapping:", train_data.class_indices)


# ==========================================================
# 3. COMPUTE CLASS WEIGHTS (IMPORTANT)
# ==========================================================

class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_data.classes),
    y=train_data.classes
)

class_weights = dict(enumerate(class_weights))
print("\n🔷 CLASS WEIGHTS APPLIED:", class_weights)


# ==========================================================
# 4. MODEL SETUP — RESNET50 (FINE TUNING ENABLED)
# ==========================================================

base_model = ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3))

# UNFREEZE LAST 50 LAYERS FOR FINE-TUNING
base_model.trainable = True
for layer in base_model.layers[:100]:
    layer.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.4),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(NUM_CLASSES, activation='softmax')  # MULTI-CLASS OUTPUT
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()


# ==========================================================
# 5. TRAINING SETUP
# ==========================================================

checkpoint = ModelCheckpoint(
    "models/best_model.h5",
    save_best_only=True,
    monitor='val_loss',
    mode='min'
)

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

# ==========================================================
# 6. TRAIN MODEL (NOW CORRECT)
# ==========================================================

history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS,
    callbacks=[checkpoint, early_stop],
    class_weight=class_weights  # <<—— IMPORTANT
)


# ==========================================================
# 7. SAVE FINAL MODEL
# ==========================================================

os.makedirs("models", exist_ok=True)
model.save("models/lung_cancer_model.h5")
print("\n🎉 Model saved successfully to models/lung_cancer_model.h5")


# ==========================================================
# 8. PLOTTING RESULTS
# ==========================================================

plt.figure()
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title("Accuracy Over Time")
plt.show()

plt.figure()
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title("Loss Over Time")
plt.show()
