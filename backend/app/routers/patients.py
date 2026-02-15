from fastapi import APIRouter, Depends, HTTPException, File, UploadFile
from sqlalchemy.orm import Session
from typing import List
from database.database import get_db
from database import crud
from shared.schemas import PatientCreate, PatientResponse
from backend.app.config import get_settings
import os
import uuid
from pathlib import Path

router = APIRouter()
settings = get_settings()


@router.post("/", response_model=PatientResponse)
def create_patient(patient: PatientCreate, db: Session = Depends(get_db)):
    """Create a new patient"""
    db_patient = crud.create_patient(
        db=db,
        first_name=patient.first_name,
        last_name=patient.last_name,
        age=patient.age,
        sex=patient.sex.value,
        location=patient.location,
        chief_complaint=patient.chief_complaint,
        clinical_history=patient.clinical_history,
    )
    return db_patient


@router.get("/", response_model=List[PatientResponse])
def list_patients(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    """List all patients"""
    patients = crud.get_patients(db, skip=skip, limit=limit)
    return patients


@router.get("/{patient_id}", response_model=PatientResponse)
def get_patient(patient_id: str, db: Session = Depends(get_db)):
    """Get patient by ID"""
    patient = crud.get_patient(db, patient_id)
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    return patient


@router.get("/case/{case_id}", response_model=PatientResponse)
def get_patient_by_case(case_id: str, db: Session = Depends(get_db)):
    """Get patient by case ID"""
    patient = crud.get_patient_by_case_id(db, case_id)
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    return patient


@router.put("/{patient_id}", response_model=PatientResponse)
def update_patient(
    patient_id: str, patient_update: PatientCreate, db: Session = Depends(get_db)
):
    """Update patient information"""
    patient = crud.update_patient(
        db=db,
        patient_id=patient_id,
        first_name=patient_update.first_name,
        last_name=patient_update.last_name,
        age=patient_update.age,
        sex=patient_update.sex.value,
        location=patient_update.location,
        chief_complaint=patient_update.chief_complaint,
        clinical_history=patient_update.clinical_history,
    )
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    return patient


@router.post("/{patient_id}/upload-history-pdf")
async def upload_clinical_history_pdf(
    patient_id: str,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    """Upload a PDF for patient clinical history. Extracts text and stores both."""
    # Validate patient exists
    patient = crud.get_patient(db, patient_id)
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")

    # Validate file type
    file_ext = Path(file.filename).suffix.lower()
    if file_ext != ".pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")

    # Read file
    contents = await file.read()

    # Check size (10MB max)
    if len(contents) > settings.MAX_FILE_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Maximum: {settings.MAX_FILE_SIZE / (1024*1024):.0f}MB",
        )

    # Save file
    file_id = str(uuid.uuid4())
    new_filename = f"{patient_id}_clinical_history_{file_id}.pdf"
    file_path = os.path.join(settings.UPLOAD_DIR, new_filename)

    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    with open(file_path, "wb") as f:
        f.write(contents)

    # Extract text from PDF
    extracted_text = ""
    try:
        from PyPDF2 import PdfReader

        reader = PdfReader(file_path)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                extracted_text += page_text + "\n"
        extracted_text = extracted_text.strip()
    except Exception as e:
        extracted_text = f"(PDF uploaded but text extraction failed: {str(e)})"

    # Update patient record
    patient.clinical_history_pdf = file_path
    if extracted_text:
        existing = patient.clinical_history or ""
        if existing:
            patient.clinical_history = (
                existing + "\n\n--- Extracted from PDF ---\n" + extracted_text
            )
        else:
            patient.clinical_history = extracted_text
    db.commit()
    db.refresh(patient)

    return {
        "message": "Clinical history PDF uploaded successfully",
        "patient_id": patient_id,
        "pdf_path": file_path,
        "extracted_text_length": len(extracted_text),
        "clinical_history": patient.clinical_history,
    }
