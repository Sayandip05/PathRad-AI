from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from database.database import get_db
from database import crud
from shared.schemas import ReportGenerateRequest, ReportResponse
from shared.utils.pdf_generator import generate_diagnosis_report
from shared.utils.text_report_generator import save_text_report
from backend.app.config import get_settings
from datetime import datetime
import os

router = APIRouter()
settings = get_settings()


@router.post("/generate", response_model=ReportResponse)
def generate_report(
    request: ReportGenerateRequest,
    format: str = "pdf",  # pdf or txt
    db: Session = Depends(get_db),
):
    """Generate report for a diagnosis (PDF or TXT format)"""
    diagnosis = crud.get_diagnosis(db, request.diagnosis_id)
    if not diagnosis:
        raise HTTPException(status_code=404, detail="Diagnosis not found")

    if not diagnosis.patient:
        raise HTTPException(status_code=404, detail="Patient not found")

    # Generate report based on format
    if format.lower() == "txt":
        report_path = save_text_report(
            diagnosis=diagnosis,
            patient=diagnosis.patient,
            output_dir=settings.REPORTS_DIR,
        )
        media_type = "text/plain"
    else:  # default to PDF
        report_path = generate_diagnosis_report(
            diagnosis=diagnosis,
            patient=diagnosis.patient,
            include_images=request.include_images,
            output_dir=settings.REPORTS_DIR,
        )
        media_type = "application/pdf"

    # Update diagnosis with report path
    crud.update_diagnosis(db, request.diagnosis_id, report_path=report_path)

    # Create download URL
    filename = os.path.basename(report_path)
    download_url = f"/api/reports/download/{filename}"

    return ReportResponse(
        report_id=diagnosis.id,
        diagnosis_id=diagnosis.id,
        report_path=report_path,
        download_url=download_url,
        generated_at=datetime.utcnow(),
    )


@router.get("/download/{filename}")
def download_report(filename: str):
    """Download generated report (auto-detects PDF or TXT)"""
    file_path = os.path.join(settings.REPORTS_DIR, filename)

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Report not found")

    # Determine media type from extension
    if filename.endswith(".txt"):
        media_type = "text/plain"
    else:
        media_type = "application/pdf"

    return FileResponse(file_path, media_type=media_type, filename=filename)


@router.get("/diagnosis/{diagnosis_id}")
def get_diagnosis_report(diagnosis_id: str, db: Session = Depends(get_db)):
    """Get report info for a diagnosis"""
    diagnosis = crud.get_diagnosis(db, diagnosis_id)
    if not diagnosis:
        raise HTTPException(status_code=404, detail="Diagnosis not found")

    if not diagnosis.report_path or not os.path.exists(diagnosis.report_path):
        raise HTTPException(status_code=404, detail="Report not generated yet")

    filename = os.path.basename(diagnosis.report_path)

    return {
        "diagnosis_id": diagnosis_id,
        "report_url": f"/api/reports/download/{filename}",
        "generated_at": diagnosis.completed_at.isoformat()
        if diagnosis.completed_at
        else None,
    }
