from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
from typing import List, Optional
from database.database import get_db
from database import crud
from shared.schemas import (
    DiagnosisCreate,
    DiagnosisResponse,
    DiagnosisDetail,
    TriageResult,
    RadiologyResult,
    PathologyResult,
    ClinicalContext,
)
import sys
import os

# Add agents to path
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
)
from agents.orchestrator_agent.production_orchestrator import run_diagnosis_workflow

router = APIRouter()


@router.post("/", response_model=DiagnosisResponse)
def create_diagnosis(
    diagnosis: DiagnosisCreate,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
):
    """Create a new diagnosis request - triggers full production pipeline"""
    # Verify patient exists
    patient = crud.get_patient(db, diagnosis.patient_id)
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")

    # Create diagnosis record
    db_diagnosis = crud.create_diagnosis(db, diagnosis.patient_id)

    # Get image paths from database
    xray_path = None
    microscopy_path = None

    if diagnosis.xray_image_id:
        xray_image = crud.get_medical_image(db, diagnosis.xray_image_id)
        if xray_image:
            xray_path = xray_image.file_path

    if diagnosis.microscopy_image_id:
        micro_image = crud.get_medical_image(db, diagnosis.microscopy_image_id)
        if micro_image:
            microscopy_path = micro_image.file_path

    # Run production pipeline in background with real model inference
    background_tasks.add_task(
        run_diagnosis_workflow,
        patient_id=diagnosis.patient_id,
        diagnosis_id=db_diagnosis.id,
        xray_path=xray_path,
        microscopy_path=microscopy_path,
    )

    return db_diagnosis


@router.get("/", response_model=List[DiagnosisResponse])
def list_diagnoses(skip: int = 0, limit: int = 50, db: Session = Depends(get_db)):
    """List recent diagnoses"""
    diagnoses = crud.get_recent_diagnoses(db, limit=limit)
    return diagnoses


@router.get("/{diagnosis_id}", response_model=DiagnosisDetail)
def get_diagnosis(diagnosis_id: str, db: Session = Depends(get_db)):
    """Get diagnosis by ID with full details"""
    diagnosis = crud.get_diagnosis(db, diagnosis_id)
    if not diagnosis:
        raise HTTPException(status_code=404, detail="Diagnosis not found")

    # Use DiagnosisResponse to properly resolve ORM relationships (patient etc.)
    base = DiagnosisResponse.model_validate(diagnosis)
    base_dict = base.model_dump()

    # Safely parse agent JSON fields — if schema mismatch, store as None
    parsed_triage = None
    if diagnosis.triage_result and isinstance(diagnosis.triage_result, dict):
        try:
            parsed_triage = TriageResult(**diagnosis.triage_result)
        except Exception:
            parsed_triage = None

    parsed_radiology = None
    if diagnosis.radiology_result and isinstance(diagnosis.radiology_result, dict):
        try:
            parsed_radiology = RadiologyResult(**diagnosis.radiology_result)
        except Exception:
            parsed_radiology = None

    parsed_pathology = None
    if diagnosis.pathology_details and isinstance(diagnosis.pathology_details, dict):
        try:
            parsed_pathology = PathologyResult(**diagnosis.pathology_details)
        except Exception:
            parsed_pathology = None

    parsed_clinical = None
    if diagnosis.clinical_context and isinstance(diagnosis.clinical_context, dict):
        try:
            parsed_clinical = ClinicalContext(**diagnosis.clinical_context)
        except Exception:
            parsed_clinical = None

    return DiagnosisDetail(
        **base_dict,
        triage_result=parsed_triage,
        radiology_result=parsed_radiology,
        pathology_details=parsed_pathology,
        clinical_context=parsed_clinical,
    )


@router.get("/patient/{patient_id}", response_model=List[DiagnosisResponse])
def get_patient_diagnoses(patient_id: str, db: Session = Depends(get_db)):
    """Get all diagnoses for a patient"""
    diagnoses = crud.get_patient_diagnoses(db, patient_id)
    return diagnoses


@router.post("/{diagnosis_id}/retry")
def retry_diagnosis(
    diagnosis_id: str, background_tasks: BackgroundTasks, db: Session = Depends(get_db)
):
    """Retry a failed diagnosis with full pipeline"""
    diagnosis = crud.get_diagnosis(db, diagnosis_id)
    if not diagnosis:
        raise HTTPException(status_code=404, detail="Diagnosis not found")

    # Reset status
    crud.update_diagnosis(db, diagnosis_id, status="pending")

    # Get image paths
    xray_path = None
    microscopy_path = None

    images = crud.get_patient_images(db, diagnosis.patient_id)
    for img in images:
        if img.image_type == "xray":
            xray_path = img.file_path
        elif img.image_type == "microscopy":
            microscopy_path = img.file_path

    # Re-run full production pipeline
    background_tasks.add_task(
        run_diagnosis_workflow,
        patient_id=diagnosis.patient_id,
        diagnosis_id=diagnosis_id,
        xray_path=xray_path,
        microscopy_path=microscopy_path,
    )

    return {
        "message": "Diagnosis retry initiated with full pipeline",
        "diagnosis_id": diagnosis_id,
    }
