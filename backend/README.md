# Backend — FastAPI Server

## File Guide

| File | Purpose | Open to explain |
|------|---------|-----------------|
| `main.py` | **FastAPI API server** — all endpoints (`/predict`, `/explain`, `/evaluate`, `/model-info`, `/history`). Start here. | ✅ Primary file |
| `train.py` | **Model training** — MobileNetV2 base, 2-phase fine-tuning, L2 regularization, Dropout, BatchNorm, callbacks | ✅ Show for architecture/training |
| `evaluate.py` | **Evaluation script** — generates confusion matrix, per-class F1/Precision/Recall, saves `eval_results.json` | ✅ Show for performance metrics |
| `model_checkpoint.h5` | **Saved model weights** — loaded by `main.py` at startup (13 MB) | — Binary, don't open |
| `eval_results.json` | **Pre-computed metrics** — served by `/evaluate` endpoint | ✅ Show the JSON output |
| `venv/` | Python virtual environment — dependencies only | — Don't open |
| `_archive/` | Scratch/debug scripts used during development — not needed for running | — Skip |

---

## Running

```bash
# Activate env and start server
.\venv\Scripts\python.exe main.py

# Re-run evaluation (updates eval_results.json)
.\venv\Scripts\python.exe evaluate.py

# Re-train model from scratch
.\venv\Scripts\python.exe train.py
```

---

## Key Design Decisions (for explanation)

- **MobileNetV2** chosen for efficiency (3.4M params vs ResNet's 25M) while achieving >89% accuracy
- **2-phase training**: Phase 1 trains only the custom head; Phase 2 unfreezes last 30 layers for fine-tuning
- **Grad-CAM** uses intermediate conv layer gradients to highlight pixels that drove the prediction
- **L2 + Dropout** prevent overfitting on the 6-class imbalanced dataset
