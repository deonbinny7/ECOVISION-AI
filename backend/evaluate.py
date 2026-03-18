"""
EcoVision AI — Model Evaluation Script
=======================================
Course  : MAI417-3 Deep Learning
Exam    : NeuralHack 2026 — CHRIST (Deemed to be University)

Run this ONCE after training to generate:
  - eval_results.json  (confusion matrix, per-class metrics)
  - These results are served by the /evaluate API endpoint

Usage:
    cd backend
    python evaluate.py
"""

import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score
)

CLASSES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
BASE_DIR = r"C:\Users\deonb\OneDrive\Desktop\ESE\GARBAGE CLASSIFICATION"
TEST_DIR = os.path.join(BASE_DIR, 'TEST')
TARGET_SIZE = (224, 224)
BATCH_SIZE = 32


def run_evaluation():
    print("[INFO] Loading model...")
    model_path = 'model.h5' if os.path.exists('model.h5') else 'model_checkpoint.h5'
    model = tf.keras.models.load_model(model_path)

    test_datagen = ImageDataGenerator(rescale=1.0 / 255)
    test_generator = test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=TARGET_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )

    print("[INFO] Running inference on test set...")
    predictions = model.predict(test_generator, verbose=1)
    y_pred = np.argmax(predictions, axis=1)
    y_true = test_generator.classes

    # ── Metrics ──────────────────────────────────
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    cm = confusion_matrix(y_true, y_pred).tolist()

    report = classification_report(
        y_true, y_pred,
        target_names=CLASSES,
        output_dict=True,
        zero_division=0
    )

    # Per-class metrics
    per_class = {}
    for cls in CLASSES:
        if cls in report:
            per_class[cls] = {
                "precision": round(report[cls]["precision"], 4),
                "recall": round(report[cls]["recall"], 4),
                "f1_score": round(report[cls]["f1-score"], 4),
                "support": int(report[cls]["support"])
            }

    results = {
        "overall": {
            "accuracy": round(acc, 4),
            "weighted_precision": round(prec, 4),
            "weighted_recall": round(rec, 4),
            "weighted_f1": round(f1, 4),
            "num_test_samples": int(len(y_true))
        },
        "confusion_matrix": cm,
        "class_labels": CLASSES,
        "per_class_metrics": per_class
    }

    with open('eval_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"[RESULT] Test Accuracy  : {acc:.4f}")
    print(f"[RESULT] Weighted F1    : {f1:.4f}")
    print(f"[RESULT] Saved eval_results.json")
    return results


if __name__ == "__main__":
    run_evaluation()
