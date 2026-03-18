# EcoVision AI — Garbage Classification
## CHRIST (Deemed to be University), Bangalore
### MAI417-3 Deep Learning | NeuralHack 2026 | MSAIM III Trimester

---

## 1. Problem Definition (CO1 — L3)

### 1.1 Real-World Problem
Global waste misclassification leads to approximately **91% of plastic waste** not being recycled correctly. Manual sorting is labour-intensive, error-prone, and unsustainable at scale. An intelligent waste classification system can automate this process, increasing recycling rates and reducing landfill burden.

### 1.2 Learning Task
| Component | Description |
|-----------|-------------|
| **Task Type** | Multi-class supervised image classification |
| **Input** | RGB image of a waste item (224 × 224 × 3) |
| **Output** | Probability distribution over 6 classes → argmax → predicted class |
| **Classes** | `cardboard`, `glass`, `metal`, `paper`, `plastic`, `trash` |
| **Dataset** | Garbage Classification Dataset (TRAIN/TEST split) |

### 1.3 Real-World Relevance
This system can be deployed in:
- Smart recycling bins with camera sensors
- Municipal waste sorting conveyor belts
- Mobile applications for household waste guidance

---

## 2. Mathematical Modeling (CO2 — L4)

### 2.1 Architecture: Convolutional Neural Network (CNN)

A CNN applies a series of learnable filters across the spatial input:

```
Conv output: (f * I)(x, y) = Σᵢ Σⱼ I(x+i, y+j) · f(i, j)
```

We use **MobileNetV2** (transfer learning) as the feature extractor — a depthwise separable CNN pre-trained on ImageNet. The custom classification head maps extracted features to class probabilities.

### 2.2 Complete Architecture

| Layer | Output Shape | Parameters | Notes |
|-------|-------------|-----------|-------|
| MobileNetV2 (frozen Phase 1) | 7×7×1280 | 2,257,984 | Pre-trained backbone |
| GlobalAveragePooling2D | 1280 | 0 | Spatial pooling |
| BatchNormalization | 1280 | 5,120 | Stabilise activations |
| Dense(256, ReLU) | 256 | 327,936 | L2(λ=0.001) |
| Dropout(0.4) | 256 | 0 | Regularisation |
| Dense(128, ReLU) | 128 | 32,896 | L2(λ=0.001) |
| Dropout(0.3) | 128 | 0 | Regularisation |
| Dense(6, Softmax) | 6 | 774 | Output layer |

### 2.3 Activation Functions

**Hidden layers — ReLU:**
```
f(x) = max(0, x)
```
Introduces non-linearity and avoids the vanishing gradient problem.

**Output layer — Softmax:**
```
σ(z)ᵢ = e^{zᵢ} / Σⱼ e^{zⱼ}     for i = 1, ..., K  (K = 6)
```
Converts raw logits into a probability distribution summing to 1.

### 2.4 Loss Function — Categorical Cross-Entropy

```
L(y, ŷ) = -(1/N) Σᵢ₌₁ᴺ Σₖ₌₁ᴷ yᵢₖ · log(ŷᵢₖ)
```

Where:
- `yᵢₖ` = ground truth (one-hot vector: 1 if sample i belongs to class k)
- `ŷᵢₖ` = predicted probability for sample i, class k
- `N`   = number of training samples
- `K`   = number of classes (6)

### 2.5 Regularization

**L2 Weight Regularization:**
```
L_total = L_CE + λ · Σⱼ ||Wⱼ||²     (λ = 0.001)
```
Penalises large weights, preventing overfitting.

**Dropout:**
Each neuron output is set to zero with probability `p` during training:
```
ỹ = y · Bernoulli(1 - p) / (1 - p)
```
Applied with `p = 0.4` (first Dropout) and `p = 0.3` (second Dropout).

### 2.6 Optimizer — Adam

Adam combines momentum and adaptive learning rates:
```
mₜ = β₁·mₜ₋₁ + (1 - β₁)·gₜ             (1st moment — momentum)
vₜ = β₂·vₜ₋₁ + (1 - β₂)·gₜ²            (2nd moment — adaptive scale)
m̂ₜ = mₜ / (1 - β₁ᵗ)                    (bias correction)
v̂ₜ = vₜ / (1 - β₂ᵗ)                    (bias correction)
θₜ₊₁ = θₜ - α · m̂ₜ / (√v̂ₜ + ε)        (parameter update)
```

Hyperparameters: `α = 1e-4` (Phase 1), `α = 1e-5` (Phase 2), `β₁ = 0.9`, `β₂ = 0.999`, `ε = 1e-8`

### 2.7 Evaluation Metrics

| Metric | Formula | Purpose |
|--------|---------|---------|
| Accuracy | TP+TN / (TP+TN+FP+FN) | Overall correctness |
| Precision | TP / (TP + FP) | Correctness of positive predictions |
| Recall | TP / (TP + FN) | Coverage of actual positives |
| F1-Score | 2·P·R / (P + R) | Harmonic mean, handles imbalance |
| Confusion Matrix | K×K matrix | Per-class error analysis |

---

## 3. Model Design and Implementation (CO3 — L5)

### 3.1 Transfer Learning Strategy

MobileNetV2 was chosen for:
- Lightweight architecture (2.2M params vs. 25M for ResNet50)
- Depthwise separable convolutions — efficient for deployment
- Strong ImageNet features transferable to garbage textures

### 3.2 Two-Phase Training Strategy

**Phase 1: Feature Extraction** (Epochs 1–15)
- Freeze all MobileNetV2 layers
- Train only the custom Dense head
- Adam lr = 1e-4

**Phase 2: Fine-tuning** (Epochs 1–10)
- Unfreeze last 30 layers of MobileNetV2
- Lower lr = 1e-5 to preserve pre-trained weights
- Catastrophic forgetting avoided by low LR

### 3.3 Data Augmentation Pipeline

| Transform | Parameter | Purpose |
|-----------|-----------|---------|
| Rescale | ÷255 | Normalise to [0,1] |
| Rotation | ±30° | Rotation invariance |
| Width/Height shift | ±20% | Translation invariance |
| Zoom | ±20% | Scale invariance |
| Horizontal flip | True | Mirror augmentation |
| Shear | 0.2 | Perspective variation |

---

## 4. Data Preparation and Training (CO4 — L5)

- **Dataset**: Garbage Classification — 6 classes, ~2500 training images
- **Preprocessing**: Resize to 224×224, normalise to [0,1]
- **Validation**: Held-out TEST split (no augmentation applied)
- **Reproducibility**: Fixed seeds (NumPy=42, TensorFlow=42, Python=42)
- **Callbacks**:
  - `EarlyStopping` — stops training if val_accuracy doesn't improve for 5 epochs
  - `ReduceLROnPlateau` — reduces LR by 0.2× if val_loss plateaus for 3 epochs
  - `ModelCheckpoint` — saves the best model by val_accuracy

---

## 5. User Interface (CO5 — L6)

The EcoVision AI web interface (Next.js + FastAPI) provides:

1. **Classify Tab**: Upload a waste image → instant prediction with Grad-CAM heatmap overlay, probability bar chart across all 6 classes, disposal advice, and CO₂ impact info
2. **Evaluate Tab**: Interactive confusion matrix, per-class Precision / Recall / F1 table
3. **Model Info Tab**: Architecture diagram, mathematical formulation, hyperparameters

### Grad-CAM Explainability
Grad-CAM visualises which regions of the image the model focused on:
```
αₖ = (1/Z) Σᵢⱼ ∂yᶜ/∂Aᵢⱼᵏ       (global avg pool of gradients)
Lgrad-cam = ReLU(Σₖ αₖ · Aᵏ)    (weighted conv feature maps)
```

---

## 6. Innovation and Real-World Relevance

- **Grad-CAM Explainability**: Makes the model's decisions interpretable — critical for trust in AI systems
- **CO₂ Savings Feedback**: Quantifies environmental impact of correct sorting per material
- **Disposal Optimization Engine**: Provides class-specific recycling guidance
- **Two-phase fine-tuning**: Efficiently adapts a large pre-trained model with minimal compute
- **Production-ready API**: FastAPI backend with structured JSON responses for easy integration

---

*End of Report — EcoVision AI | MAI417-3 Deep Learning | NeuralHack 2026*
