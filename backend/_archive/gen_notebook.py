import json

def make_code_cell(code):
    lines = [line + "\n" for line in code.strip().split("\n")]
    if lines:
        lines[-1] = lines[-1].rstrip("\n")
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": lines
    }

def make_markdown_cell(text):
    lines = [line + "\n" for line in text.strip().split("\n")]
    if lines:
        lines[-1] = lines[-1].rstrip("\n")
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": lines
    }

cells = []

cells.append(make_markdown_cell("""
# EcoVision AI — Deep Learning Pipeline (Google Colab Version)
======================================================
Course  : MAI417-3 Deep Learning (MSAIM III Trimester)
Exam    : NeuralHack 2026 — CHRIST (Deemed to be University)

### Instructions for Colab:
1. Zip your dataset folder `GARBAGE CLASSIFICATION` into `dataset.zip`.
2. Upload `dataset.zip` to your Google Colab session files.
3. Run the cells sequentially! Make sure you are using a T4 GPU runtime (Runtime > Change runtime type).
"""))

cells.append(make_code_cell("""
# 1. Unzip dataset (Run this once you have uploaded dataset.zip)
!unzip -q dataset.zip -d ./
"""))

cells.append(make_code_cell("""
import os
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.regularizers import l2
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from PIL import Image

# Reproducibility Seed
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

CLASSES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
NUM_CLASSES = len(CLASSES)
"""))

cells.append(make_markdown_cell("""
### 1. Data Loading & Augmentation
"""))

cells.append(make_code_cell("""
# Update these paths based on how your zip extracts
base_dir = './GARBAGE CLASSIFICATION'
train_dir = os.path.join(base_dir, 'TRAIN')
validation_dir = os.path.join(base_dir, 'TEST')  # Using TEST as validation

TARGET_SIZE = (224, 224)
BATCH_SIZE = 32

train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
validation_datagen = ImageDataGenerator(rescale=1.0 / 255)

print("[INFO] Loading training data...")
train_generator = train_datagen.flow_from_directory(
    train_dir, target_size=TARGET_SIZE, batch_size=BATCH_SIZE, class_mode='categorical', shuffle=True, seed=SEED
)

print("[INFO] Loading testing data...")
test_generator = validation_datagen.flow_from_directory(
    validation_dir, target_size=TARGET_SIZE, batch_size=BATCH_SIZE, class_mode='categorical', shuffle=False
)
"""))

cells.append(make_markdown_cell("""
### 2. Model Architecture
"""))

cells.append(make_code_cell("""
def create_model():
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False  # Freeze Phase 1

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

    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=1e-4), metrics=['accuracy'])
    return model

model = create_model()
model.summary()
"""))

cells.append(make_markdown_cell("""
### 3. Phase 1 Training (Transfer Learning - Custom Head)
"""))

cells.append(make_code_cell("""
early_stop = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6, verbose=1)
checkpoint = ModelCheckpoint('model_checkpoint.h5', monitor='val_accuracy', save_best_only=True, verbose=1)

print("[INFO] Phase 1: Training classification head...")
history1 = model.fit(
    train_generator,
    epochs=15, 
    validation_data=test_generator,
    callbacks=[early_stop, reduce_lr, checkpoint]
)
"""))

cells.append(make_markdown_cell("""
### 4. Phase 2 Training (Fine-tuning Top Base Layers)
"""))

cells.append(make_code_cell("""
print("[INFO] Phase 2: Fine-tuning base model...")
base_model = model.layers[0]
base_model.trainable = True
# Unfreeze last 30 layers
for layer in base_model.layers[:-30]:
    layer.trainable = False

model.compile(optimizer=Adam(learning_rate=1e-5), loss='categorical_crossentropy', metrics=['accuracy'])

history2 = model.fit(
    train_generator,
    epochs=10,
    validation_data=test_generator,
    callbacks=[early_stop, reduce_lr, checkpoint]
)
"""))

cells.append(make_markdown_cell("""
### 5. Plot Training History
"""))

cells.append(make_code_cell("""
acc = history1.history['accuracy'] + history2.history['accuracy']
val_acc = history1.history['val_accuracy'] + history2.history['val_accuracy']
loss = history1.history['loss'] + history2.history['loss']
val_loss = history1.history['val_loss'] + history2.history['val_loss']

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(acc, label='Train Acc')
plt.plot(val_acc, label='Val Acc')
plt.axvline(x=len(history1.history['accuracy'])-1, color='red', linestyle='--', label='Phase 2 Start')
plt.legend()
plt.title('Accuracy')

plt.subplot(1, 2, 2)
plt.plot(loss, label='Train Loss')
plt.plot(val_loss, label='Val Loss')
plt.axvline(x=len(history1.history['loss'])-1, color='red', linestyle='--', label='Phase 2 Start')
plt.legend()
plt.title('Loss')
plt.show()
"""))

cells.append(make_markdown_cell("""
### 6. Evaluation (Confusion Matrix & Classification Report)
"""))

cells.append(make_code_cell("""
print("[INFO] Running inference on test set...")
predictions = model.predict(test_generator, verbose=1)
y_pred = np.argmax(predictions, axis=1)
y_true = test_generator.classes

print("\\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=test_generator.class_indices.keys()))

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=test_generator.class_indices.keys(), yticklabels=test_generator.class_indices.keys())
plt.ylabel('True Class')
plt.xlabel('Predicted Class')
plt.title('Confusion Matrix')
plt.show()
"""))

cells.append(make_markdown_cell("""
### 7. Grad-CAM Explainability
"""))

cells.append(make_code_cell("""
def generate_gradcam(model, img_array, class_idx):
    base_model = model.layers[0]
    last_conv_name = None
    for layer in reversed(base_model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv_name = layer.name
            break

    conv_layer = base_model.get_layer(last_conv_name)
    conv_model = tf.keras.Model(inputs=base_model.input, outputs=conv_layer.output)

    with tf.GradientTape() as tape:
        conv_outputs = conv_model(img_array)
        tape.watch(conv_outputs)
        
        x = conv_outputs
        found = False
        for layer in base_model.layers:
            if layer.name == last_conv_name:
                found = True
                continue
            if found:
                x = layer(x)
                
        for layer in model.layers[1:]:
            x = layer(x, training=False) if hasattr(layer, 'training') else layer(x)
        predictions = x
        loss = predictions[:, class_idx]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = conv_outputs[0] @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    heatmap = heatmap.numpy()

    heatmap_resized = cv2.resize(heatmap, (224, 224))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    return colored

# Test Grad-CAM on one random image from the test set
batch_x, batch_y = next(test_generator)
img_array = batch_x[0:1] # Take first image
true_label = np.argmax(batch_y[0])

preds = model.predict(img_array, verbose=0)
pred_label = np.argmax(preds[0])

heatmap = generate_gradcam(model, img_array, pred_label)

# Overlay
orig_img = np.uint8(img_array[0] * 255)
orig_bgr = cv2.cvtColor(orig_img, cv2.COLOR_RGB2BGR)
overlay = cv2.addWeighted(orig_bgr, 0.55, heatmap, 0.45, 0)
overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(img_array[0])
plt.title(f"Original (True: {CLASSES[true_label]})")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(overlay_rgb)
plt.title(f"Grad-CAM (Pred: {CLASSES[pred_label]})")
plt.axis('off')
plt.show()
"""))

notebook = {
    "cells": cells,
    "metadata": {
        "colab": {"provenance": []},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"}
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

with open(r"C:\Users\deonb\OneDrive\Desktop\ESE\dataset\EcoVision_Colab_Complete.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2)

print("Notebook generated successfully!")
