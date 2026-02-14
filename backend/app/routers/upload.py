from fastapi import APIRouter, File, UploadFile, HTTPException, Depends
from sqlalchemy.orm import Session
from database.database import get_db
from database import crud
from shared.schemas import ImageUploadResponse
from shared.utils.image_processing import process_uploaded_image, assess_image_quality
from backend.app.config import get_settings
import os
import uuid
from pathlib import Path

router = APIRouter()
settings = get_settings()

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".dcm"}
MAX_FILE_SIZE = settings.MAX_FILE_SIZE  # 10MB


@router.post("/{patient_id}/{image_type}", response_model=ImageUploadResponse)
async def upload_image(
    patient_id: str,
    image_type: str,  # 'xray' or 'microscopy'
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    """Upload medical image for a patient"""

    # Validate patient exists
    patient = crud.get_patient(db, patient_id)
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")

    # Validate image type
    if image_type not in ["xray", "microscopy"]:
        raise HTTPException(
            status_code=400, detail="Invalid image type. Must be 'xray' or 'microscopy'"
        )

    # Validate file extension
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}",
        )

    # Read file content
    contents = await file.read()

    # Check file size
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Maximum size: {MAX_FILE_SIZE / (1024 * 1024):.1f}MB",
        )

    # Generate unique filename
    file_id = str(uuid.uuid4())
    new_filename = f"{patient_id}_{image_type}_{file_id}{file_ext}"
    file_path = os.path.join(settings.UPLOAD_DIR, new_filename)

    # Save file
    with open(file_path, "wb") as f:
        f.write(contents)

    # Process image
    try:
        processed_image = process_uploaded_image(file_path)
        quality_metrics = assess_image_quality(processed_image)

        # Create database record
        image_record = crud.create_medical_image(
            db=db,
            patient_id=patient_id,
            image_type=image_type,
            file_path=file_path,
            original_filename=file.filename,
            quality_score=quality_metrics.get("quality_score"),
            brightness=quality_metrics.get("brightness"),
            contrast=quality_metrics.get("contrast"),
            sharpness=quality_metrics.get("sharpness"),
            is_gradable=quality_metrics.get("adequate", True),
        )

        message = "Image uploaded successfully"
        if not quality_metrics.get("adequate", True):
            issues = quality_metrics.get("issues", [])
            message = f"Image uploaded but quality issues detected: {', '.join(issues)}"

        return ImageUploadResponse(
            image_id=image_record.id,
            patient_id=patient_id,
            image_type=image_type,
            quality_score=quality_metrics.get("quality_score"),
            is_gradable=quality_metrics.get("adequate", True),
            message=message,
        )

    except Exception as e:
        # Clean up file if processing failed
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(
            status_code=500, detail=f"Image processing failed: {str(e)}"
        )


@router.get("/{image_id}")
def get_image_info(image_id: str, db: Session = Depends(get_db)):
    """Get information about an uploaded image"""
    image = crud.get_medical_image(db, image_id)
    if not image:
        raise HTTPException(status_code=404, detail="Image not found")

    return {
        "id": image.id,
        "patient_id": image.patient_id,
        "image_type": image.image_type,
        "original_filename": image.original_filename,
        "quality_score": image.quality_score,
        "is_gradable": image.is_gradable,
        "created_at": image.created_at.isoformat(),
    }


@router.delete("/{image_id}")
def delete_image(image_id: str, db: Session = Depends(get_db)):
    """Delete an uploaded image"""
    image = crud.get_medical_image(db, image_id)
    if not image:
        raise HTTPException(status_code=404, detail="Image not found")

    # Delete file from filesystem
    if os.path.exists(image.file_path):
        os.remove(image.file_path)

    # Delete from database
    db.delete(image)
    db.commit()

    return {"message": "Image deleted successfully"}
