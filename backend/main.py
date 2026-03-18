"""
EcoVision AI — FastAPI Backend
================================
Course  : MAI417-3 Deep Learning (MSAIM III Trimester)
Exam    : NeuralHack 2026 — CHRIST (Deemed to be University)

Endpoints:
  POST /predict     — Classify an uploaded image
  POST /explain     — Generate Grad-CAM heatmap for an uploaded image
  GET  /evaluate    — Return confusion matrix & per-class metrics (pre-computed)
  GET  /model-info  — Return architecture, loss fn, optimizer, regularisation info
  GET  /history     — Return training loss/accuracy history
"""

import os
import io
import json
import base64
import numpy as np
import tensorflow as tf
import cv2
from PIL import Image
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

app = FastAPI(title="EcoVision AI", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────────
# Model Loading
# ─────────────────────────────────────────────────
model = None
try:
    for path in ["model.h5", "model_checkpoint.h5"]:
        if os.path.exists(path):
            model = tf.keras.models.load_model(path)
            print(f"[INFO] Model loaded from {path}")
            break
except Exception as e:
    print(f"[WARN] Could not load model: {e}")

CLASSES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

DISPOSAL_TIPS = {
    'cardboard': 'Flatten and place in the Paper/Cardboard recycling bin.',
    'glass':     'Rinse and place in the Glass recycling bin. Do not break.',
    'metal':     'Rinse cans and place in the Mixed Recycling bin.',
    'paper':     'Dry paper goes in the Paper recycling bin.',
    'plastic':   'Check the recycling number; most soft plastics go in Mixed Recycling.',
    'trash':     'Non-recyclable. Place in General Waste bin.',
}

CO2_SAVINGS = {
    'cardboard': '1.1 kg CO₂ saved per kg recycled',
    'glass':     '0.3 kg CO₂ saved per kg recycled',
    'metal':     '4.0 kg CO₂ saved per kg recycled',
    'paper':     '0.9 kg CO₂ saved per kg recycled',
    'plastic':   '1.5 kg CO₂ saved per kg recycled',
    'trash':     'Minimal savings — reduce single-use consumption',
}


def preprocess_image(contents: bytes):
    """Resize, normalise, and expand dims for model input."""
    img = Image.open(io.BytesIO(contents)).convert('RGB')
    img = img.resize((224, 224))
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0), img


# ─────────────────────────────────────────────────
# Grad-CAM Helper
# Visualises which spatial regions drove the prediction
# ─────────────────────────────────────────────────
def generate_gradcam(model, img_array, class_idx):
    """
    Gradient-weighted Class Activation Mapping (Grad-CAM).
    Works with the existing Sequential model (MobileNetV2 base).

    Grad-CAM formula:
      α_k = (1/Z) Σ_{i,j} ∂y^c / ∂A^k_{i,j}   (global avg of gradients)
      L^c_Grad-CAM = ReLU( Σ_k α_k · A^k )      (weighted feature maps)
    """
    try:
        base_model = model.layers[0]  # MobileNetV2 Functional model

        # Find the last Conv2D layer name in MobileNetV2
        last_conv_name = None
        for layer in reversed(base_model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                last_conv_name = layer.name
                break

        if last_conv_name is None:
            return None

        # Build a sub-model from MobileNetV2's input to last conv output
        conv_layer = base_model.get_layer(last_conv_name)
        conv_model = tf.keras.Model(
            inputs=base_model.input,
            outputs=conv_layer.output
        )

        # Watch the conv output to compute gradients
        with tf.GradientTape() as tape:
            conv_outputs = conv_model(img_array)
            tape.watch(conv_outputs)
            # Now run the rest of the sequential model from layer 1 onward
            x = conv_outputs
            # Apply remaining MobileNetV2 layers after the conv layer
            found = False
            for layer in base_model.layers:
                if layer.name == last_conv_name:
                    found = True
                    continue
                if found:
                    x = layer(x)
            # Now apply custom head layers (GlobalAveragePooling, Dense, Dropout, Dense)
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

        # Resize to 224×224 and apply JET colormap
        heatmap_resized = cv2.resize(heatmap, (224, 224))
        heatmap_uint8 = np.uint8(255 * heatmap_resized)
        colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        return colored
    except Exception as e:
        print(f"[WARN] Grad-CAM failed: {e}")
        return None


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    POST /predict
    Accepts an image file and returns the predicted garbage class,
    confidence score, all class probabilities, disposal tip, and CO2 info.
    """
    if not model:
        return {"error": "Model not loaded. Run train.py first.", "status": "fail"}

    contents = await file.read()
    img_array, _ = preprocess_image(contents)
    predictions = model.predict(img_array, verbose=0)
    probs = predictions[0].tolist()
    pred_idx = int(np.argmax(probs))

    return {
        "class": CLASSES[pred_idx],
        "confidence": round(probs[pred_idx], 4),
        "all_probabilities": {cls: round(p, 4) for cls, p in zip(CLASSES, probs)},
        "disposal_tip": DISPOSAL_TIPS[CLASSES[pred_idx]],
        "co2_impact": CO2_SAVINGS[CLASSES[pred_idx]],
    }


@app.post("/explain")
async def explain(file: UploadFile = File(...)):
    """
    POST /explain
    Returns a base64-encoded Grad-CAM heatmap overlaid on the input image,
    highlighting regions that most influenced the model's prediction.
    """
    if not model:
        return {"error": "Model not loaded.", "status": "fail"}

    contents = await file.read()
    img_array, pil_img = preprocess_image(contents)
    predictions = model.predict(img_array, verbose=0)
    pred_idx = int(np.argmax(predictions[0]))

    heatmap = generate_gradcam(model, img_array, pred_idx)

    # Overlay heatmap on original image
    orig_np = np.array(pil_img)  # already 224x224 from preprocess
    orig_bgr = cv2.cvtColor(orig_np, cv2.COLOR_RGB2BGR)
    overlay = cv2.addWeighted(orig_bgr, 0.55, heatmap, 0.45, 0)
    overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

    _, buffer = cv2.imencode('.png', overlay_rgb)
    b64 = base64.b64encode(buffer).decode('utf-8')

    return {
        "class": CLASSES[pred_idx],
        "confidence": round(float(predictions[0][pred_idx]), 4),
        "gradcam_image": f"data:image/png;base64,{b64}"
    }


@app.get("/evaluate")
async def evaluate():
    """
    GET /evaluate
    Returns pre-computed evaluation metrics (confusion matrix, per-class F1).
    Run evaluate.py once after training to generate eval_results.json.
    """
    if not os.path.exists('eval_results.json'):
        return {
            "error": "eval_results.json not found. Run evaluate.py first.",
            "hint": "cd backend && python evaluate.py"
        }
    with open('eval_results.json', 'r') as f:
        return json.load(f)


@app.get("/history")
async def history():
    """
    GET /history
    Returns training accuracy/loss curve data for visualisation.
    """
    if not os.path.exists('training_history.json'):
        return {"error": "training_history.json not found. Run train.py first."}
    with open('training_history.json', 'r') as f:
        return json.load(f)


@app.get("/model-info")
async def model_info():
    """
    GET /model-info
    Returns architecture summary, math formulation, and training hyper-parameters.
    """
    layers_info = []
    if model:
        for layer in model.layers:
            try:
                out_shape = str(layer.output_shape)
            except AttributeError:
                out_shape = "N/A"
            layers_info.append({
                "name": layer.name,
                "type": layer.__class__.__name__,
                "trainable": layer.trainable,
                "output_shape": out_shape
            })

    # Build dynamic custom head description from actual loaded model
    head_layers = []
    if model:
        for layer in model.layers[1:]:  # Skip MobileNetV2 base
            name = layer.__class__.__name__
            cfg = layer.get_config()
            if name == "Dense":
                reg = cfg.get("kernel_regularizer")
                reg_str = " + L2(λ=0.001)" if reg else ""
                head_layers.append(f"Dense({cfg.get('units')}, {cfg.get('activation','linear').upper()}){reg_str}")
            elif name == "Dropout":
                head_layers.append(f"Dropout(p={cfg.get('rate')})")
            elif name == "GlobalAveragePooling2D":
                head_layers.append("GlobalAveragePooling2D")
            elif name == "BatchNormalization":
                head_layers.append("BatchNormalization")
            else:
                head_layers.append(name)
    if not head_layers:
        head_layers = [
            "GlobalAveragePooling2D",
            "Dense(256, ReLU) + L2(λ=0.001)",
            "Dropout(p=0.4)",
            "Dense(6, Softmax)"
        ]

    return {
        "model_name": "EcoVision CNN (MobileNetV2 Transfer Learning)",
        "task": "Multi-class Image Classification",
        "classes": CLASSES,
        "num_classes": len(CLASSES),
        "input_shape": [224, 224, 3],
        "architecture": {
            "base": "MobileNetV2 (pre-trained on ImageNet)",
            "custom_head": head_layers
        },
        "math_formulation": {
            "activation_hidden": "ReLU: f(x) = max(0, x)",
            "activation_output": "Softmax: σ(z)ᵢ = eᶻⁱ / Σⱼ eᶻʲ",
            "loss_function": "Categorical Cross-Entropy: L = -Σᵢ Σₖ yᵢₖ · log(ŷᵢₖ)",
            "optimizer": "Adam: θₜ₊₁ = θₜ - α·m̂ₜ / (√v̂ₜ + ε)",
            "l2_regularization": "L2: L_total = L_CE + λ·||W||²",
            "dropout": "Randomly zeroes p fraction of neurons during training",
        },
        "hyperparameters": {
            "phase1_lr": 1e-4,
            "phase2_lr": 1e-5,
            "batch_size": 32,
            "phase1_epochs": 15,
            "phase2_epochs": 10,
            "l2_lambda": 0.001,
            "dropout_rates": [0.4, 0.3],
            "seed": 42
        },
        "training_strategy": {
            "phase1": "Freeze MobileNetV2; train Dense head only",
            "phase2": "Unfreeze last 30 layers; fine-tune with LR=1e-5",
            "callbacks": ["EarlyStopping(patience=5)", "ReduceLROnPlateau(factor=0.2)", "ModelCheckpoint"]
        }
    }


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
