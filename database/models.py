from sqlalchemy import (
    Column,
    Integer,
    String,
    Float,
    DateTime,
    Text,
    Boolean,
    ForeignKey,
    JSON,
)
from sqlalchemy.orm import relationship
from database.database import Base
from datetime import datetime
import uuid


def generate_uuid():
    return str(uuid.uuid4())


class Patient(Base):
    __tablename__ = "patients"

    id = Column(String, primary_key=True, default=generate_uuid)
    case_id = Column(
        String,
        unique=True,
        index=True,
        default=lambda: f"CASE-{datetime.now().strftime('%Y%m%d')}-{generate_uuid()[:8].upper()}",
    )

    # Demographics
    age = Column(Integer, nullable=False)
    sex = Column(String(10), nullable=False)

    # Clinical Information
    chief_complaint = Column(Text)
    clinical_history = Column(Text)

    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    diagnoses = relationship(
        "Diagnosis", back_populates="patient", cascade="all, delete-orphan"
    )
    images = relationship(
        "MedicalImage", back_populates="patient", cascade="all, delete-orphan"
    )


class MedicalImage(Base):
    __tablename__ = "medical_images"

    id = Column(String, primary_key=True, default=generate_uuid)
    patient_id = Column(String, ForeignKey("patients.id"))

    # Image Info
    image_type = Column(String(50))  # 'xray', 'microscopy'
    file_path = Column(String(500))
    original_filename = Column(String(255))

    # Image Quality Metrics
    quality_score = Column(Float)
    brightness = Column(Float)
    contrast = Column(Float)
    sharpness = Column(Float)
    is_gradable = Column(Boolean, default=True)

    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    patient = relationship("Patient", back_populates="images")


class Diagnosis(Base):
    __tablename__ = "diagnoses"

    id = Column(String, primary_key=True, default=generate_uuid)
    patient_id = Column(String, ForeignKey("patients.id"))

    # Diagnosis Results
    primary_diagnosis = Column(Text)
    confidence = Column(Float)
    urgency_level = Column(String(20))  # 'critical', 'high', 'medium', 'low'

    # Detailed Findings
    findings = Column(JSON)  # List of findings
    differential_diagnoses = Column(JSON)  # List of alternatives

    # TB Specific
    tb_probability = Column(Float)
    tb_risk_score = Column(Float)

    # Pathology Results
    pathology_result = Column(String(100))  # 'Negative', 'Positive', 'Scanty', etc.
    bacilli_count = Column(Integer)

    # Agent Results
    triage_result = Column(JSON)
    radiology_result = Column(JSON)
    pathology_details = Column(JSON)
    clinical_context = Column(JSON)

    # Treatment & Follow-up
    treatment_plan = Column(Text)
    follow_up = Column(Text)
    human_review_required = Column(Boolean, default=False)

    # Report
    report_path = Column(String(500))

    # Status
    status = Column(
        String(50), default="pending"
    )  # pending, processing, completed, error

    created_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime)

    # Relationships
    patient = relationship("Patient", back_populates="diagnoses")


class AgentLog(Base):
    __tablename__ = "agent_logs"

    id = Column(String, primary_key=True, default=generate_uuid)
    diagnosis_id = Column(String, ForeignKey("diagnoses.id"))

    agent_name = Column(
        String(50)
    )  # 'triage', 'radiologist', 'pathologist', 'clinical', 'orchestrator'
    action = Column(String(100))
    input_data = Column(JSON)
    output_data = Column(JSON)
    execution_time_ms = Column(Float)

    created_at = Column(DateTime, default=datetime.utcnow)
