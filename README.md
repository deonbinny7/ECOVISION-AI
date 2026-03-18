# EcoVision AI 🌱

An end-to-end intelligent waste classification system powered by deep learning. EcoVision AI instantly classifies waste, provides Grad-CAM visual explainability, and offers tailored recycling guidance to promote sustainable disposal practices.

**Developed for MAI417-3 | MSAIM III Trimester | NeuralHack 2026**

![EcoVision Banner](https://via.placeholder.com/1200x400/2A3B32/FFFFFF?text=EcoVision+AI+-+Intelligent+Garbage+Classification) *(You can replace this placeholder banner with an actual screenshot of the app)*

## 🌟 Key Features

- **Instant Classification:** Identifies 6 classes of waste (`cardboard`, `glass`, `metal`, `paper`, `plastic`, `trash`) using a pre-trained **MobileNetV2** model.
- **Explainable AI (XAI):** Generates **Grad-CAM heatmaps** to show exactly which parts of the image influenced the model's decision, building trust in the AI.
- **Sustainability Focus:** Provides immediate, class-specific recycling instructions and CO₂ savings feedback to the user.
- **Performance Evaluation:** Full transparency with an interactive confusion matrix and detailed per-class metrics.

## 🏗️ Architecture

The system follows a modern decoupled architecture:

- **Frontend:** Next.js & React (TypeScript, Tailwind CSS) — A visually engaging, interactive UI.
- **Backend:** FastAPI (Python) — High-performance REST API serving the deep learning model.
- **Model:** MobileNetV2 (Transfer Learning) — Fine-tuned in a two-phase strategy for optimal accuracy without overfitting.

## 🚀 Getting Started

Follow these instructions to set up and run the project locally.

### Prerequisites

- Node.js (v18+)
- Python (3.10+)
- pip & venv

### 1. Clone the Repository

```bash
git clone <your-repository-url>
cd ESE
```

### 2. Start the Backend (FastAPI)

The backend handles the model inference and evaluation endpoints.

```bash
# Navigate to the backend directory
cd backend

# Create a virtual environment (if not already done)
python -m venv venv

# Activate the virtual environment
# On Windows:
.\venv\Scripts\activate
# On macOS/Linux:
# source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the API server
python main.py
```

The API will be available at `http://localhost:8000`.

### 3. Start the Frontend (Next.js)

The frontend provides the user interface for classification and evaluation.

```bash
# Navigate to the frontend directory
cd frontend

# Install dependencies
npm install

# Start the development server
npm run dev
```

The UI will be available at `http://localhost:3000`.

### 4. Initialise Evaluation Data (First Run Only)

To enable the "Evaluate" tab in the UI, you must generate the initial evaluation results. With the virtual environment activated in the `backend` folder, run:

```bash
cd backend
python evaluate.py
```
*(This writes `eval_results.json` which the frontend expects)*

---

## 📊 Endpoints & Architecture Detail

### Core API Endpoints

- `POST /predict`: Classifies an uploaded image and returns prediction, confidence, and disposal tips.
- `POST /explain`: Generates and returns a Grad-CAM heatmap.
- `GET /evaluate`: Returns confusion matrix and per-class metrics.
- `GET /model-info`: Provides model mathematical formulation and layer details.

### Deep Learning Pipeline

- **Transfer Learning:** MobileNetV2 backbone (frozen initially, then fine-tuned).
- **Custom Head:** Global Average Pooling → Dense(256) → Dropout(0.4) → Dense(128) → Dropout(0.3) → Softmax(6).
- **Optimization:** Adam Optimizer, Categorical Cross-Entropy Loss with L2 Regularization.

For a deeper dive into the methodology, mathematics, and metrics, please refer to the detailed [REPORT.md](./REPORT.md).

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the issues page.

## 📄 License

This project is created for academic purposes.
