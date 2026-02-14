from sqlalchemy.orm import Session
from database.models import Patient, MedicalImage, Diagnosis, AgentLog
from typing import Optional, List
from datetime import datetime


# Patient CRUD
def create_patient(
    db: Session,
    age: int,
    sex: str,
    chief_complaint: Optional[str] = None,
    clinical_history: Optional[str] = None,
) -> Patient:
    """Create a new patient record"""
    patient = Patient(
        age=age,
        sex=sex,
        chief_complaint=chief_complaint,
        clinical_history=clinical_history,
    )
    db.add(patient)
    db.commit()
    db.refresh(patient)
    return patient


def get_patient(db: Session, patient_id: str) -> Optional[Patient]:
    """Get patient by ID"""
    return db.query(Patient).filter(Patient.id == patient_id).first()


def get_patient_by_case_id(db: Session, case_id: str) -> Optional[Patient]:
    """Get patient by case ID"""
    return db.query(Patient).filter(Patient.case_id == case_id).first()


def get_patients(db: Session, skip: int = 0, limit: int = 100) -> List[Patient]:
    """Get list of patients with pagination"""
    return (
        db.query(Patient)
        .order_by(Patient.created_at.desc())
        .offset(skip)
        .limit(limit)
        .all()
    )


def update_patient(db: Session, patient_id: str, **kwargs) -> Optional[Patient]:
    """Update patient information"""
    patient = get_patient(db, patient_id)
    if patient:
        for key, value in kwargs.items():
            if hasattr(patient, key):
                setattr(patient, key, value)
        patient.updated_at = datetime.utcnow()
        db.commit()
        db.refresh(patient)
    return patient


# Medical Image CRUD
def create_medical_image(
    db: Session,
    patient_id: str,
    image_type: str,
    file_path: str,
    original_filename: str,
    quality_score: Optional[float] = None,
    brightness: Optional[float] = None,
    contrast: Optional[float] = None,
    sharpness: Optional[float] = None,
    is_gradable: bool = True,
) -> MedicalImage:
    """Create medical image record"""
    image = MedicalImage(
        patient_id=patient_id,
        image_type=image_type,
        file_path=file_path,
        original_filename=original_filename,
        quality_score=quality_score,
        brightness=brightness,
        contrast=contrast,
        sharpness=sharpness,
        is_gradable=is_gradable,
    )
    db.add(image)
    db.commit()
    db.refresh(image)
    return image


def get_medical_image(db: Session, image_id: str) -> Optional[MedicalImage]:
    """Get medical image by ID"""
    return db.query(MedicalImage).filter(MedicalImage.id == image_id).first()


def get_patient_images(
    db: Session, patient_id: str, image_type: Optional[str] = None
) -> List[MedicalImage]:
    """Get all images for a patient"""
    query = db.query(MedicalImage).filter(MedicalImage.patient_id == patient_id)
    if image_type:
        query = query.filter(MedicalImage.image_type == image_type)
    return query.all()


# Diagnosis CRUD
def create_diagnosis(db: Session, patient_id: str) -> Diagnosis:
    """Create a new diagnosis record"""
    diagnosis = Diagnosis(patient_id=patient_id, status="pending")
    db.add(diagnosis)
    db.commit()
    db.refresh(diagnosis)
    return diagnosis


def get_diagnosis(db: Session, diagnosis_id: str) -> Optional[Diagnosis]:
    """Get diagnosis by ID"""
    return db.query(Diagnosis).filter(Diagnosis.id == diagnosis_id).first()


def get_patient_diagnoses(db: Session, patient_id: str) -> List[Diagnosis]:
    """Get all diagnoses for a patient"""
    return (
        db.query(Diagnosis)
        .filter(Diagnosis.patient_id == patient_id)
        .order_by(Diagnosis.created_at.desc())
        .all()
    )


def update_diagnosis(db: Session, diagnosis_id: str, **kwargs) -> Optional[Diagnosis]:
    """Update diagnosis with results"""
    diagnosis = get_diagnosis(db, diagnosis_id)
    if diagnosis:
        for key, value in kwargs.items():
            if hasattr(diagnosis, key):
                setattr(diagnosis, key, value)

        if kwargs.get("status") == "completed":
            diagnosis.completed_at = datetime.utcnow()

        db.commit()
        db.refresh(diagnosis)
    return diagnosis


def get_recent_diagnoses(db: Session, limit: int = 50) -> List[Diagnosis]:
    """Get recent diagnoses"""
    return db.query(Diagnosis).order_by(Diagnosis.created_at.desc()).limit(limit).all()


# Agent Log CRUD
def create_agent_log(
    db: Session,
    diagnosis_id: str,
    agent_name: str,
    action: str,
    input_data: Optional[dict] = None,
    output_data: Optional[dict] = None,
    execution_time_ms: Optional[float] = None,
) -> AgentLog:
    """Log agent activity"""
    log = AgentLog(
        diagnosis_id=diagnosis_id,
        agent_name=agent_name,
        action=action,
        input_data=input_data or {},
        output_data=output_data or {},
        execution_time_ms=execution_time_ms,
    )
    db.add(log)
    db.commit()
    db.refresh(log)
    return log


def get_diagnosis_logs(db: Session, diagnosis_id: str) -> List[AgentLog]:
    """Get all logs for a diagnosis"""
    return (
        db.query(AgentLog)
        .filter(AgentLog.diagnosis_id == diagnosis_id)
        .order_by(AgentLog.created_at)
        .all()
    )


# Statistics
def get_statistics(db: Session) -> dict:
    """Get system statistics"""
    total_patients = db.query(Patient).count()
    total_diagnoses = db.query(Diagnosis).count()
    completed_diagnoses = (
        db.query(Diagnosis).filter(Diagnosis.status == "completed").count()
    )
    pending_diagnoses = (
        db.query(Diagnosis).filter(Diagnosis.status == "pending").count()
    )
    tb_detected = db.query(Diagnosis).filter(Diagnosis.tb_probability > 0.7).count()

    return {
        "total_patients": total_patients,
        "total_diagnoses": total_diagnoses,
        "completed_diagnoses": completed_diagnoses,
        "pending_diagnoses": pending_diagnoses,
        "tb_detected": tb_detected,
    }
