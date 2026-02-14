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

    # Parse stored JSON fields
    diagnosis_detail = DiagnosisDetail(
        **diagnosis.__dict__,
        triage_result=TriageResult(**diagnosis.triage_result)
        if diagnosis.triage_result
        else None,
        radiology_result=RadiologyResult(**diagnosis.radiology_result)
        if diagnosis.radiology_result
        else None,
        pathology_details=PathologyResult(**diagnosis.pathology_details)
        if diagnosis.pathology_details
        else None,
        clinical_context=ClinicalContext(**diagnosis.clinical_context)
        if diagnosis.clinical_context
        else None,
    )

    return diagnosis_detail


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
