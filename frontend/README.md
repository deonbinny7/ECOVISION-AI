# Frontend — Next.js UI

## File Guide

### Pages (`src/app/`)

| File | Purpose |
|------|---------|
| `page.tsx` | **Root page** — tab navigation (Classify / Evaluate / Model Info), app layout |
| `layout.tsx` | Next.js root layout — global font, metadata |
| `globals.css` | Global styles — glassmorphism cards, glow effects, animations |

### Components (`src/components/`)

| File | Purpose | Open to explain |
|------|---------|-----------------|
| `Dashboard.tsx` | **Main classify tab** — image upload, calls `/predict` + `/explain` in parallel, shows probability bars, Grad-CAM toggle, disposal tips | ✅ Primary component |
| `EvalPanel.tsx` | **Evaluate tab** — fetches `/evaluate`, renders confusion matrix + per-class metrics table | ✅ Show for performance results |
| `ModelInfo.tsx` | **Model Info tab** — fetches `/model-info`, renders architecture layers, math formulas, hyperparameters | ✅ Show for architecture explanation |
| `PipelineAnimation.tsx` | Animated pipeline stages shown during inference (Upload → Preprocess → CNN → Softmax → Result) | ✅ Show for visual explanation |
| `ScrollSequence.tsx` | Animated hero background scroll effect on initial load | — Visual only |

---

## Running

```bash
npm run dev
# → http://localhost:3000
```

---

## Architecture Flow (for explanation)

```
User uploads image
    ↓
Dashboard.tsx fires two parallel fetch() calls:
    ├── POST /predict  →  class label + probabilities + disposal tip
    └── POST /explain  →  base64 Grad-CAM heatmap
    ↓
PipelineAnimation.tsx shows 8-stage processing animation
    ↓
Results rendered: classification card + probability bars + Grad-CAM toggle
```
