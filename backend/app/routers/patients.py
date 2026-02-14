from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List
from database.database import get_db
from database import crud
from shared.schemas import PatientCreate, PatientResponse

router = APIRouter()


@router.post("/", response_model=PatientResponse)
def create_patient(patient: PatientCreate, db: Session = Depends(get_db)):
    """Create a new patient"""
    db_patient = crud.create_patient(
        db=db,
        age=patient.age,
        sex=patient.sex.value,
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
        age=patient_update.age,
        sex=patient_update.sex.value,
        chief_complaint=patient_update.chief_complaint,
        clinical_history=patient_update.clinical_history,
    )
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    return patient
