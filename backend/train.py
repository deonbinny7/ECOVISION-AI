"""
EcoVision AI — Garbage Classification Training Script
======================================================
Course  : MAI417-3 Deep Learning (MSAIM III Trimester)
Exam    : NeuralHack 2026 — CHRIST (Deemed to be University)
Author  : EcoVision AI Team

Architecture  : MobileNetV2 (CNN Transfer Learning) + Custom Classification Head
Task Type     : Multi-class Image Classification (6 categories)
Loss Function : Categorical Cross-Entropy
              : L(y, ŷ) = -Σ y_i · log(ŷ_i)
Optimizer     : Adam (Adaptive Moment Estimation)
              : θ_{t+1} = θ_t - α · m̂_t / (√v̂_t + ε)
Regularization: Dropout (p=0.4) + L2 Weight Decay (λ=0.001)
Metrics       : Accuracy, Validation Loss
"""

import os
import json
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.regularizers import l2

# ──────────────────────────────────────────────
# Reproducibility Seed (CO2 — Evaluation Metric)
# ──────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ──────────────────────────────────────────────
# Class Definitions
# 6 garbage categories derived from the dataset
# ──────────────────────────────────────────────
CLASSES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
NUM_CLASSES = len(CLASSES)


def create_model():
    """
    Build the CNN Classification Model.

    Architecture:
    ─────────────────────────────────────────────────────
     Layer                    Output Shape    Params
    ─────────────────────────────────────────────────────
     MobileNetV2 (frozen)     7×7×1280        2,257,984
     GlobalAveragePooling2D   1280            0
     BatchNormalization        1280            5,120
     Dense(256, relu) + L2    256             327,936
     Dropout(0.4)             256             0
     Dense(128, relu) + L2    128             32,896
     Dropout(0.3)             128             0
     Dense(6, softmax)         6              774
    ─────────────────────────────────────────────────────

    Activation Functions:
      - Hidden layers : ReLU  → f(x) = max(0, x)
      - Output layer  : Softmax → σ(z)_i = e^{z_i} / Σ_j e^{z_j}

    Regularization:
      - L2: adds λ·||W||² to loss → penalises large weights
      - Dropout: randomly zeroes p fraction of neurons during training
    """
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3)
    )
    base_model.trainable = False  # Phase 1: Freeze base CNN

    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        BatchNormalization(),
        Dense(256, activation='relu', kernel_regularizer=l2(0.001)),
        Dropout(0.4),
        Dense(128, activation='relu', kernel_regularizer=l2(0.001)),
        Dropout(0.3),
        Dense(NUM_CLASSES, activation='softmax')
    ], name='EcoVision_CNN')

    # ────────────────────────────────────────────
    # Loss: Categorical Cross-Entropy
    #   L = -1/N Σ_i Σ_k y_{i,k} · log(ŷ_{i,k})
    # Optimizer: Adam  lr=1e-4
    # ────────────────────────────────────────────
    model.compile(
        loss='categorical_crossentropy',
        optimizer=Adam(learning_rate=1e-4),
        metrics=['accuracy']
    )
    return model


def main():
    base_dir = r"C:\Users\deonb\OneDrive\Desktop\ESE\GARBAGE CLASSIFICATION"
    train_dir = os.path.join(base_dir, 'TRAIN')
    validation_dir = os.path.join(base_dir, 'TEST')

    TARGET_SIZE = (224, 224)
    BATCH_SIZE = 32

    # ──────────────────────────────────────────
    # Data Augmentation (CO3 — Data Preparation)
    # Artificially expands training set to improve
    # generalisation and reduce overfitting.
    # ──────────────────────────────────────────
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,          # Normalize pixel values to [0, 1]
        rotation_range=30,           # Random rotation ±30°
        width_shift_range=0.2,       # Horizontal shift
        height_shift_range=0.2,      # Vertical shift
        shear_range=0.2,             # Shear transformation
        zoom_range=0.2,              # Random zoom
        horizontal_flip=True,        # Mirror augmentation
        fill_mode='nearest'          # Fill strategy for new pixels
    )
    validation_datagen = ImageDataGenerator(rescale=1.0 / 255)

    print("[INFO] Loading training data...")
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=TARGET_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=True,
        seed=SEED
    )

    print("[INFO] Loading validation data...")
    validation_generator = validation_datagen.flow_from_directory(
        validation_dir,
        target_size=TARGET_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )

    # Save class indices for consistent label mapping
    class_indices = train_generator.class_indices
    with open('class_indices.json', 'w') as f:
        json.dump(class_indices, f, indent=2)
    print(f"[INFO] Class indices: {class_indices}")

    model = create_model()
    model.summary()

    # ──────────────────────────────────────────
    # Callbacks
    # ──────────────────────────────────────────
    early_stop = EarlyStopping(
        monitor='val_accuracy',
        patience=5,
        restore_best_weights=True,
        verbose=1
    )
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=3,
        min_lr=1e-6,
        verbose=1
    )
    checkpoint = ModelCheckpoint(
        'model_checkpoint.h5',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )

    # ──────────────────────────────────────────────
    # Phase 1: Train only the custom classification head
    # Base MobileNetV2 layers are frozen (transfer learning)
    # ──────────────────────────────────────────────
    print("[INFO] Phase 1: Training classification head...")
    history1 = model.fit(
        train_generator,
        epochs=15,
        validation_data=validation_generator,
        callbacks=[early_stop, reduce_lr, checkpoint]
    )
    model.save('model_phase1.h5')

    # ──────────────────────────────────────────────
    # Phase 2: Fine-tuning — unfreeze last 30 conv layers
    # Lower LR (1e-5) to avoid catastrophic forgetting
    # ──────────────────────────────────────────────
    print("[INFO] Phase 2: Fine-tuning base model...")
    base_model = model.layers[0]
    base_model.trainable = True
    for layer in base_model.layers[:-30]:
        layer.trainable = False

    model.compile(
        optimizer=Adam(learning_rate=1e-5),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    history2 = model.fit(
        train_generator,
        epochs=10,
        validation_data=validation_generator,
        callbacks=[early_stop, reduce_lr, checkpoint]
    )
    model.save('model.h5')

    # ──────────────────────────────────────────────
    # Export full training history for UI visualisation
    # Saved as training_history.json
    # ──────────────────────────────────────────────
    def combine_histories(h1, h2):
        combined = {}
        for key in h1.history:
            combined[key] = h1.history[key] + h2.history[key]
        return combined

    full_history = combine_histories(history1, history2)
    with open('training_history.json', 'w') as f:
        json.dump(full_history, f, indent=2)

    print("[INFO] Training complete. Model saved to model.h5")
    print(f"[INFO] Final val_accuracy: {max(full_history.get('val_accuracy', [0])):.4f}")


if __name__ == "__main__":
    main()
