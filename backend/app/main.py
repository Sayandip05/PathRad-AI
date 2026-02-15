from fastapi import (
    FastAPI,
    File,
    UploadFile,
    HTTPException,
    Depends,
    BackgroundTasks,
    WebSocket,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
import os
import uuid
from datetime import datetime
from typing import Optional, List

from backend.app.config import get_settings
from database.database import engine, Base, get_db
from database import crud
from models import model_manager
from shared.schemas import (
    PatientCreate,
    PatientResponse,
    ImageUploadResponse,
    DiagnosisCreate,
    DiagnosisResponse,
    DiagnosisDetail,
    ReportGenerateRequest,
    ReportResponse,
)
from backend.app.websocket_manager import websocket_endpoint
from agents.orchestrator_agent.production_orchestrator import run_diagnosis_workflow

# Create database tables
Base.metadata.create_all(bind=engine)

# Initialize FastAPI app
app = FastAPI(
    title="PathRad AI API - Production",
    description="Production Multi-Agent Diagnostic System with Real-Time Updates",
    version="2.0.0",
)

# Get settings
settings = get_settings()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static directories
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
os.makedirs(settings.REPORTS_DIR, exist_ok=True)
app.mount("/uploads", StaticFiles(directory=settings.UPLOAD_DIR), name="uploads")
app.mount("/reports", StaticFiles(directory=settings.REPORTS_DIR), name="reports")


# Initialize models on startup
@app.on_event("startup")
async def startup_event():
    """Initialize ML models on startup with GPU enforcement"""
    print("\n" + "=" * 70)
    print("🚀 PATHRAD AI PRODUCTION SYSTEM STARTING")
    print("=" * 70)
    print("\n📋 Initializing ML Models on GPU...")
    print("   This may take a few minutes...")

    try:
        # Verify GPU availability
        import torch

        if torch.cuda.is_available():
            print(f"\n✅ GPU Detected: {torch.cuda.get_device_name(0)}")
            print(f"   CUDA Version: {torch.version.cuda}")
            print(
                f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
            )
        else:
            print("\n⚠️  WARNING: GPU not detected! System will run on CPU (slow)")

        # Initialize models
        model_manager.initialize(
            medgemma_path=settings.MEDGEMMA_MODEL_PATH,
            cxr_path=settings.CXR_MODEL_PATH,
            path_path=settings.PATH_MODEL_PATH,
            hf_token=settings.HF_TOKEN,
        )

        print("\n" + "=" * 70)
        print("✅ SYSTEM READY - All Models Loaded")
        print("=" * 70)

    except Exception as e:
        print(f"\n❌ CRITICAL ERROR: Could not initialize models: {e}")
        print("   System may not function correctly without models")
        raise


# WebSocket endpoint for real-time updates
@app.websocket("/ws/{patient_id}")
async def websocket_route(websocket: WebSocket, patient_id: str):
    """WebSocket endpoint for real-time pipeline updates"""
    await websocket_endpoint(websocket, patient_id)


# Include routers
from backend.app.routers import patients, diagnosis, reports, upload

app.include_router(patients.router, prefix="/api/patients", tags=["patients"])
app.include_router(diagnosis.router, prefix="/api/diagnosis", tags=["diagnosis"])
app.include_router(reports.router, prefix="/api/reports", tags=["reports"])
app.include_router(upload.router, prefix="/api/upload", tags=["upload"])


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "PathRad AI API - Production",
        "version": "2.0.0",
        "status": "operational",
        "docs": "/docs",
        "features": [
            "Real-time WebSocket updates",
            "Production orchestrator",
            "GPU model inference",
            "Confidence gates",
            "Multilingual voice input (Whisper local)",
        ],
    }


@app.get("/health")
async def health_check():
    """Health check endpoint with GPU status"""
    import torch

    gpu_info = {}
    if torch.cuda.is_available():
        gpu_info = {
            "available": True,
            "device": torch.cuda.get_device_name(0),
            "memory_total": f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB",
            "memory_allocated": f"{torch.cuda.memory_allocated(0) / 1e9:.2f} GB",
        }
    else:
        gpu_info = {"available": False}

    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "gpu": gpu_info,
        "models_loaded": {
            "medgemma": model_manager.medgemma_model is not None,
            "cxr": model_manager.cxr_model is not None,
            "path": model_manager.path_model is not None,
        },
    }


@app.get("/health/detailed")
async def health_check_detailed():
    """Detailed health check with per-model and per-agent status"""
    import torch

    # GPU info
    gpu_info = {}
    if torch.cuda.is_available():
        gpu_info = {
            "available": True,
            "device": torch.cuda.get_device_name(0),
            "memory_total": f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB",
            "memory_allocated": f"{torch.cuda.memory_allocated(0) / 1e9:.2f} GB",
        }
    else:
        gpu_info = {"available": False}

    # Per-model status
    medgemma_loaded = model_manager.medgemma_model is not None
    cxr_loaded = model_manager.cxr_model is not None
    path_loaded = model_manager.path_model is not None

    models = {
        "medgemma": {
            "loaded": medgemma_loaded,
            "name": "MedGemma 4B",
            "device": "cpu" if medgemma_loaded else "n/a",
            "description": "Multimodal medical AI for text generation and diagnosis",
        },
        "cxr_foundation": {
            "loaded": cxr_loaded,
            "name": "CXR Foundation",
            "device": "gpu" if cxr_loaded and torch.cuda.is_available() else ("cpu" if cxr_loaded else "n/a"),
            "description": "Chest X-ray feature extraction and analysis",
        },
        "path_foundation": {
            "loaded": path_loaded,
            "name": "Path Foundation",
            "device": "cpu" if path_loaded else "n/a",
            "description": "Pathology/microscopy image analysis",
        },
    }

    # Per-agent status (derived from model availability)
    agent_deps = {
        "triage": ["cxr_foundation"],
        "radiologist": ["cxr_foundation", "medgemma"],
        "pathologist": ["path_foundation"],
        "clinical_context": [],
        "orchestrator": [],
    }

    agent_descriptions = {
        "triage": "Rapid initial assessment, urgency scoring & image quality gates",
        "radiologist": "Chest X-ray analysis using CXR Foundation + MedGemma",
        "pathologist": "Microscopy / AFB smear analysis using Path Foundation",
        "clinical_context": "Patient history parsing, symptom extraction & risk scoring",
        "orchestrator": "Master coordinator — runs full pipeline & synthesizes report",
    }

    model_loaded_map = {
        "medgemma": medgemma_loaded,
        "cxr_foundation": cxr_loaded,
        "path_foundation": path_loaded,
    }

    agents = {}
    for agent_name, deps in agent_deps.items():
        all_deps_loaded = all(model_loaded_map.get(d, True) for d in deps)
        if not deps:
            status = "ready"
        elif all_deps_loaded:
            status = "ready"
        elif any(model_loaded_map.get(d, False) for d in deps):
            status = "degraded"
        else:
            status = "offline"

        agents[agent_name] = {
            "status": status,
            "description": agent_descriptions[agent_name],
            "depends_on": deps,
        }

    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "gpu": gpu_info,
        "models": models,
        "agents": agents,
    }


@app.get("/api/stats")
async def get_stats(db: Session = Depends(get_db)):
    """Get system statistics"""
    return crud.get_statistics(db)


@app.get("/api/languages")
async def get_languages():
    """Get supported languages for voice input via Whisper"""
    from shared.utils.whisper_integration import WhisperManager

    return {
        "languages": WhisperManager.get_supported_languages(),
        "note": "Languages supported for multilingual voice input via local Whisper model",
        "model": "whisper-base (local)",
        "total": len(WhisperManager.get_supported_languages()),
    }


@app.post("/api/transcribe")
async def transcribe_audio(
    audio: UploadFile = File(...),
    language: Optional[str] = None,
    input_type: str = "chief_complaint",
):
    """
    Transcribe audio file using local Whisper model
    Supports 99 languages for multilingual voice input
    """
    from shared.utils.whisper_integration import process_patient_voice_input
    import tempfile
    import shutil

    # Save uploaded audio to temp file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    try:
        shutil.copyfileobj(audio.file, temp_file)
        temp_file.close()

        # Transcribe using Whisper
        result = process_patient_voice_input(temp_file.name, input_type)

        if result["success"]:
            return {
                "success": True,
                "text": result["text"],
                "language": result["language"],
                "input_type": result["input_type"],
                "medical_context": result.get("medical_context", {}),
                "duration": result.get("duration", 0),
            }
        else:
            raise HTTPException(
                status_code=400, detail=result.get("error", "Transcription failed")
            )

    finally:
        # Cleanup temp file
        if os.path.exists(temp_file.name):
            os.remove(temp_file.name)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
