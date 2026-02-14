# PathRad AI — Production Multi-Agent Diagnostic System

**AI-powered medical image analysis for low-resource settings with real-time processing.**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109-009688)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-18-61DAFB)](https://reactjs.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.1-green)](https://developer.nvidia.com/cuda-downloads)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

## 📊 System Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                       PATHRAD AI SYSTEM                          │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  React Frontend ──► FastAPI Backend ──► WebSocket Real-Time      │
│   (Port 3000)        (Port 8000)        Updates                  │
│                          │                                       │
│                          ▼                                       │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │              PRODUCTION PIPELINE                          │   │
│  │  1. Image Upload → Quality Gate → Preprocessing           │   │
│  │  2. Triage Agent (CXR Features)                           │   │
│  │  3. Radiologist Agent (MedGemma)                          │   │
│  │  4. Pathologist Agent (Path Foundation)                    │   │
│  │  5. Clinical Context Agent (NLP)                          │   │
│  │  6. Confidence Gate → Treatment Plan                      │   │
│  └──────────────────────────────────────────────────────────┘   │
│                          │                                       │
│                          ▼                                       │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                    ML MODELS                              │   │
│  │  MedGemma 4B ─── CXR Foundation ─── Path Foundation      │   │
│  │  (GPU/CPU)        (CPU)              (CPU)                │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
│  Whisper (Local) ── 99 Languages ── Voice-to-Text               │
└──────────────────────────────────────────────────────────────────┘
```

---

## 🔧 Prerequisites

| Requirement     | Minimum                    | Recommended                |
|-----------------|----------------------------|----------------------------|
| **Python**      | 3.10+                      | 3.11+                      |
| **Node.js**     | 16+                        | 18+                        |
| **RAM**         | 16 GB                      | 32 GB                      |
| **GPU (VRAM)**  | Optional (runs on CPU)     | 8 GB+ (RTX 4060/4070)     |
| **Storage**     | 15 GB                      | 30 GB SSD                  |
| **OS**          | Windows 10/11, Linux       | Ubuntu 22.04 / Windows 11  |

> **Note:** If your GPU has < 8 GB VRAM (e.g., RTX 3050 4GB), MedGemma will automatically fall back to CPU with float16 precision. All features remain fully functional — inference is just slower.

---

## 🚀 Quick Start (5 Steps)

### Step 1: Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/pathrad-ai.git
cd pathrad-ai
```

### Step 2: Get a Hugging Face Token

MedGemma is a **gated model**. You need access:

1. Go to [huggingface.co/google/medgemma-4b-it](https://huggingface.co/google/medgemma-4b-it)
2. Request access (usually approved instantly)
3. Go to [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
4. Create a new token with **Read** permission
5. Copy the token (starts with `hf_...`)

### Step 3: Configure Environment

```bash
# Copy the example config
cp .env.example .env
```

Edit `.env` and set your Hugging Face token:

```env
HF_TOKEN=hf_YOUR_TOKEN_HERE
```

> All other settings have sensible defaults. See [Configuration](#-configuration) for details.

### Step 4: Backend Setup

```bash
# Create a Python virtual environment
python -m venv backend/venv

# Activate it
# Windows:
backend\venv\Scripts\activate
# Linux/macOS:
source backend/venv/bin/activate

# Install dependencies
pip install -r backend/requirements.txt

# Install PyTorch with CUDA support (for GPU acceleration)
# Choose YOUR CUDA version from: https://pytorch.org/get-started/locally/
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# If you DON'T have an NVIDIA GPU, install CPU-only PyTorch:
# pip install torch torchvision torchaudio
```

### Step 5: Frontend Setup

```bash
# Open a NEW terminal
cd frontend
npm install
```

---

## ▶️ Running the Application

### Terminal 1 — Backend

```bash
# Activate venv first
# Windows:
backend\venv\Scripts\activate
# Linux/macOS:
source backend/venv/bin/activate

# Start the backend (models download automatically on first run)
uvicorn backend.app.main:app --host 0.0.0.0 --port 8000
```

> **First run:** Models will auto-download (~10 GB total). This may take 15-30 minutes depending on your internet speed. Subsequent starts are instant.

### Terminal 2 — Frontend

```bash
cd frontend
npm start
```

### Access the Application

| Service              | URL                                   |
|----------------------|---------------------------------------|
| **Frontend (App)**   | http://localhost:3000                 |
| **Backend API**      | http://localhost:8000                 |
| **API Documentation**| http://localhost:8000/docs            |
| **Health Check**     | http://localhost:8000/health          |
| **WebSocket**        | ws://localhost:8000/ws/{patient_id}   |

---

## ✅ Pipeline Verified for New Users

This project has been thoroughly tested to ensure anyone can run it easily. Here's what happens automatically:

### 1️⃣ Clone & Configure
- Follow the [Quick Start](#-quick-start-5-steps) guide above
- Copy `.env.example` to `.env` and add your Hugging Face token
- All other settings have sensible defaults

### 2️⃣ Dependencies Handled
```bash
pip install -r backend/requirements.txt
```
- Installs all Python dependencies automatically
- Includes specific `bitsandbytes` version for 4-bit quantization
- Includes PyTorch with CUDA support (see Step 4 above)

### 3️⃣ Auto-Magic Setup

**Models** — Downloaded automatically on first run:
- ✅ MedGemma 4B (requires HF_TOKEN)
- ✅ CXR Foundation
- ✅ Path Foundation
- ✅ Whisper (base)

**Database** — Created automatically:
- SQLite database at `database/pathrad.db`
- Auto-created on first backend startup
- No manual SQL commands needed

**GPU Smart-Mode** — Automatic optimization:
| Your GPU | What Happens | Performance |
|----------|--------------|-------------|
| **RTX 4090 / 8GB+ VRAM** | Uses GPU with 4-bit quantization | ⚡ Fast |
| **RTX 3050 / < 8GB VRAM** | Falls back to CPU + float16 | 🐢 Slower but works |
| **No GPU** | Uses CPU mode | 🐢 Slower but works |

> All features remain fully functional regardless of GPU — inference speed adjusts automatically.

---

## 🧠 Models (Auto-Download)

All models download **automatically** on first startup. No manual setup required.

| Model               | Source                        | Size    | Device     | Purpose                     |
|----------------------|-------------------------------|---------|------------|-----------------------------|
| **MedGemma 4B**      | `google/medgemma-4b-it`      | ~8 GB   | GPU or CPU | Medical image analysis & QA |
| **CXR Foundation**   | `google/cxr-foundation`      | ~500 MB | CPU        | Chest X-ray features        |
| **Path Foundation**  | `google/path-foundation`     | ~150 MB | CPU        | Pathology slide analysis    |
| **Whisper** (base)   | OpenAI                        | ~150 MB | CPU        | Voice input (99 languages)  |

### GPU Memory Strategy

The system automatically selects the best loading strategy based on your GPU:

| GPU VRAM     | MedGemma Strategy       | Performance |
|--------------|-------------------------|-------------|
| **8 GB+**    | GPU, 4-bit quantization | ⚡ Fast     |
| **< 8 GB**   | CPU, float16            | 🐢 Slower  |
| **No GPU**   | CPU, float16            | 🐢 Slower  |

---

## 📁 Project Structure

```
pathrad-ai/
├── backend/                        # FastAPI Backend
│   ├── app/
│   │   ├── main.py                # Application entry point
│   │   ├── config.py              # Settings (reads .env)
│   │   ├── routers/               # API route handlers
│   │   └── websocket_manager.py   # Real-time WebSocket
│   └── requirements.txt           # Python dependencies
│
├── frontend/                       # React Frontend
│   ├── src/
│   │   ├── pages/                 # Dashboard, NewCase, Reports
│   │   ├── components/            # VoiceInput, UI components
│   │   └── services/api.js        # API client
│   └── package.json
│
├── agents/                         # Google ADK Agent Pipeline
│   ├── orchestrator_agent/        # Main pipeline coordinator
│   ├── triage_agent/              # Initial assessment
│   ├── radiologist_agent/         # Radiology analysis
│   ├── pathologist_agent/         # Pathology analysis
│   ├── clinical_context_agent/    # Clinical correlation
│   └── tools/                     # Shared agent tools
│
├── database/                       # SQLAlchemy ORM
│   ├── database.py                # DB connection & engine
│   ├── models.py                  # Data models (Patient, Diagnosis, etc.)
│   └── crud.py                    # Database operations
│
├── models/                         # ML Model Loaders
│   ├── __init__.py                # ModelManager (MedGemma, CXR, Path)
│   ├── medgemma/                  # Auto-downloaded model weights
│   ├── cxr_foundation/            # Auto-downloaded model weights
│   └── path_foundation/           # Auto-downloaded model weights
│
├── shared/                         # Shared Utilities
│   ├── schemas/                   # Pydantic request/response models
│   └── utils/
│       ├── preprocessing_pipeline.py  # Image quality gates
│       ├── confidence_gate.py         # Confidence thresholds
│       ├── pdf_generator.py           # PDF report generation
│       ├── text_report_generator.py   # Text report generation
│       └── whisper_integration.py     # Voice input (99 languages)
│
├── .env.example                    # Environment template (copy to .env)
├── .gitignore
└── README.md
```

---

## 🔧 Configuration

### Environment Variables (`.env`)

```env
# Database (auto-created in database/ folder)
DATABASE_URL=sqlite:///./database/pathrad.db

# Hugging Face Token (REQUIRED for MedGemma download)
HF_TOKEN=hf_YOUR_TOKEN_HERE

# File Storage
UPLOAD_DIR=./uploads
REPORTS_DIR=./reports

# Model Paths (auto-downloaded here)
MEDGEMMA_MODEL_PATH=./models/medgemma
CXR_MODEL_PATH=./models/cxr_foundation
PATH_MODEL_PATH=./models/path_foundation

# API Keys (Optional — for Groq LLM fallback)
GROQ_API_KEY=your-groq-api-key

# CORS (frontend origins)
ALLOWED_ORIGINS=["http://localhost:3000", "http://localhost:5173"]

# App Settings
DEBUG=true
APP_NAME=PathRad AI API
APP_VERSION=1.0.0
```

### Confidence Thresholds

Edit `shared/utils/confidence_gate.py`:

```python
CONFIDENCE_THRESHOLDS = {
    'auto_approve': 0.85,    # ≥ 0.85 → Auto-approve diagnosis
    'warning': 0.75,         # 0.75–0.85 → Warning, review suggested
    'escalate': 0.75         # < 0.75 → Escalate to human
}
```

### Image Quality Gates

Edit `shared/utils/preprocessing_pipeline.py`:

```python
QUALITY_THRESHOLDS = {
    'min_resolution': (512, 512),
    'brightness_min': 30,
    'brightness_max': 225,
    'contrast_min': 40,
    'sharpness_min': 100,
    'snr_min': 2.0
}
```

---

## 📊 API Endpoints

### Patients
| Method | Endpoint               | Description          |
|--------|------------------------|----------------------|
| POST   | `/api/patients/`       | Create new patient   |
| GET    | `/api/patients/`       | List all patients    |
| GET    | `/api/patients/{id}`   | Get patient details  |

### Diagnosis
| Method | Endpoint                      | Description               |
|--------|-------------------------------|---------------------------|
| POST   | `/api/diagnosis/`             | Create diagnosis request  |
| GET    | `/api/diagnosis/`             | List diagnoses            |
| GET    | `/api/diagnosis/{id}`         | Get diagnosis details     |
| POST   | `/api/diagnosis/{id}/retry`   | Retry failed diagnosis    |

### Image Upload
| Method | Endpoint                              | Description         |
|--------|---------------------------------------|---------------------|
| POST   | `/api/upload/{patient_id}/{type}`     | Upload medical image|

### Reports
| Method | Endpoint                              | Description         |
|--------|---------------------------------------|---------------------|
| POST   | `/api/reports/generate?format=pdf`    | Generate PDF report |
| GET    | `/api/reports/download/{filename}`    | Download report     |

### Voice Input
| Method | Endpoint            | Description                    |
|--------|---------------------|--------------------------------|
| POST   | `/api/transcribe`   | Transcribe audio (99 languages)|
| GET    | `/api/languages`    | List supported languages       |

### System
| Method | Endpoint    | Description       |
|--------|-------------|--------------------|
| GET    | `/`         | API info           |
| GET    | `/health`   | Health + model status |
| GET    | `/api/stats`| System statistics  |
| WS     | `/ws/{id}`  | Real-time updates  |

---

## 🎯 Key Features

- **Zero Mock Data** — All processing uses real ML models
- **Quality Gates** — Automatic image validation (5 quality metrics)
- **Confidence Thresholds** — ≥0.85 auto-approve, <0.75 escalate to human
- **Real-Time Updates** — WebSocket pipeline status (9 states)
- **Voice Input** — OpenAI Whisper supports 99 languages locally
- **Report Generation** — PDF and TXT formats for sharing
- **Auto Model Downloads** — All models download automatically on first run
- **GPU-Adaptive** — Automatically uses GPU if available, falls back to CPU

---

## 🧪 Verify Your Setup

### Health Check
```bash
curl http://localhost:8000/health
```

Expected response:
```json
{
  "status": "healthy",
  "gpu": {
    "available": true,
    "device": "NVIDIA GeForce RTX XXXX"
  },
  "models_loaded": {
    "medgemma": true,
    "cxr": true,
    "path": true
  }
}
```

### GPU Check
```bash
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"
```

---

## 🐛 Troubleshooting

### Models not downloading

```bash
# Verify your HF_TOKEN is set
echo $HF_TOKEN   # Linux/macOS
echo %HF_TOKEN%  # Windows

# Test token manually
python -c "from huggingface_hub import login; login(token='hf_YOUR_TOKEN')"
```

> Make sure you've accepted the MedGemma license at [huggingface.co/google/medgemma-4b-it](https://huggingface.co/google/medgemma-4b-it)

### GPU out of memory

MedGemma requires 8 GB+ VRAM for GPU mode. If you have less:
- The system **automatically falls back to CPU** — no action needed
- To force CPU mode: set `CUDA_VISIBLE_DEVICES=""` before starting

```bash
# Force CPU mode
CUDA_VISIBLE_DEVICES="" uvicorn backend.app.main:app --host 0.0.0.0 --port 8000
```

### PyTorch doesn't detect GPU

```bash
# Reinstall PyTorch with correct CUDA version
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Check CUDA version
nvidia-smi  # Shows your GPU's CUDA version
```

### Frontend won't start

```bash
cd frontend
rm -rf node_modules package-lock.json  # Clean install
npm install
npm start
```

### WebSocket connection failed

1. Verify backend is running: `curl http://localhost:8000/health`
2. Check that ports 3000 and 8000 are not blocked by firewall
3. Ensure `ALLOWED_ORIGINS` in `.env` includes your frontend URL

### Image quality check failed

- Upload images with resolution ≥ 512×512
- Ensure good lighting and focus
- Adjust thresholds in `shared/utils/preprocessing_pipeline.py`

---

## 📝 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- [Google Health AI](https://health.google/) — MedGemma, CXR Foundation, Path Foundation
- [Google ADK](https://google.github.io/adk-docs/) — Agent Development Kit
- [OpenAI Whisper](https://github.com/openai/whisper) — Multilingual speech recognition
- [FastAPI](https://fastapi.tiangolo.com/) — Backend framework
- [Material-UI](https://mui.com/) — React components

---

**Built with ❤️ for better healthcare in low-resource settings**
