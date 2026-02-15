from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum


class Sex(str, Enum):
    MALE = "male"
    FEMALE = "female"
    OTHER = "other"


class UrgencyLevel(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


# Patient Schemas
class PatientBase(BaseModel):
    first_name: str = Field(..., min_length=1, max_length=100)
    last_name: str = Field(..., min_length=1, max_length=100)
    age: int = Field(..., ge=0, le=150)
    sex: Sex
    location: Optional[str] = None
    chief_complaint: Optional[str] = None
    clinical_history: Optional[str] = None


class PatientCreate(PatientBase):
    pass


class PatientResponse(PatientBase):
    id: str
    case_id: str
    clinical_history_pdf: Optional[str] = None
    created_at: datetime

    class Config:
        from_attributes = True


# Image Schemas
class ImageUploadResponse(BaseModel):
    image_id: str
    patient_id: str
    image_type: str
    quality_score: Optional[float] = None
    is_gradable: bool
    message: str


# Diagnosis Schemas
class TriageResult(BaseModel):
    urgency_level: UrgencyLevel
    urgency_score: float = Field(..., ge=0, le=10)
    critical_flags: List[str]
    quality_assessment: Dict[str, Any]
    inference_time_ms: float


class RadiologyResult(BaseModel):
    findings: List[str]
    primary_diagnosis: str
    differential_diagnoses: List[str]
    confidence: float = Field(..., ge=0, le=1)
    tb_probability: float = Field(..., ge=0, le=1)
    localization: Dict[str, Any]


class PathologyResult(BaseModel):
    test_type: str
    result: str
    quantification: Optional[str]
    bacilli_count: int
    confidence: float = Field(..., ge=0, le=1)


class ClinicalContext(BaseModel):
    structured_history: Dict[str, Any]
    risk_scores: Dict[str, float]
    relevant_symptoms: List[str]
    comorbidities: List[str]
    risk_factors: List[str]


class DiagnosisCreate(BaseModel):
    patient_id: str
    xray_image_id: Optional[str] = None
    microscopy_image_id: Optional[str] = None


class DiagnosisResponse(BaseModel):
    id: str
    patient_id: str
    patient: PatientResponse
    primary_diagnosis: Optional[str]
    confidence: Optional[float]
    urgency_level: Optional[str]
    findings: Optional[List[str]]
    tb_probability: Optional[float]
    pathology_result: Optional[str]
    treatment_plan: Optional[str]
    follow_up: Optional[str]
    human_review_required: bool
    status: str
    report_path: Optional[str]
    created_at: datetime
    completed_at: Optional[datetime]

    class Config:
        from_attributes = True


class DiagnosisDetail(DiagnosisResponse):
    triage_result: Optional[TriageResult]
    radiology_result: Optional[RadiologyResult]
    pathology_details: Optional[PathologyResult]
    clinical_context: Optional[ClinicalContext]


# Report Schema
class ReportGenerateRequest(BaseModel):
    diagnosis_id: str
    include_images: bool = True


class ReportResponse(BaseModel):
    report_id: str
    diagnosis_id: str
    report_path: str
    download_url: str
    generated_at: datetime


# Agent Status
class AgentStatus(BaseModel):
    agent_name: str
    status: str  # 'idle', 'processing', 'error'
    last_activity: Optional[datetime]
    total_processed: int
