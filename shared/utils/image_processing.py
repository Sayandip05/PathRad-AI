import cv2
import numpy as np
from PIL import Image
from typing import Dict, Any, Tuple
import os


def process_uploaded_image(file_path: str) -> np.ndarray:
    """Load and preprocess uploaded image"""
    # Load image
    image = cv2.imread(file_path)
    if image is None:
        raise ValueError(f"Could not load image from {file_path}")

    # Convert BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image


def assess_image_quality(image: np.ndarray) -> Dict[str, Any]:
    """Assess image quality metrics"""
    # Convert to grayscale for analysis
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image

    # Calculate metrics
    brightness = np.mean(gray)
    contrast = np.std(gray)

    # Sharpness using Laplacian
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    sharpness = laplacian.var()

    # SNR
    snr = np.mean(gray) / (np.std(gray) + 1e-10)

    # Quality thresholds
    quality_thresholds = {
        "brightness_min": 30,
        "brightness_max": 225,
        "contrast_min": 40,
        "sharpness_min": 100,
    }

    # Calculate quality score
    quality_score = 0
    issues = []

    if (
        quality_thresholds["brightness_min"]
        < brightness
        < quality_thresholds["brightness_max"]
    ):
        quality_score += 25
    else:
        if brightness < quality_thresholds["brightness_min"]:
            issues.append("Image too dark")
        elif brightness > quality_thresholds["brightness_max"]:
            issues.append("Image overexposed")

    if contrast > quality_thresholds["contrast_min"]:
        quality_score += 25
    else:
        issues.append("Low contrast")

    if sharpness > quality_thresholds["sharpness_min"]:
        quality_score += 25
    else:
        issues.append("Image blurry")

    if snr > 2:
        quality_score += 25

    return {
        "quality_score": quality_score,
        "brightness": float(brightness),
        "contrast": float(contrast),
        "sharpness": float(sharpness),
        "snr": float(snr),
        "adequate": quality_score >= 60,
        "issues": issues,
    }


def preprocess_xray(
    image: np.ndarray, target_size: Tuple[int, int] = (512, 512)
) -> np.ndarray:
    """Preprocess chest X-ray for model input"""
    # Resize
    resized = cv2.resize(image, target_size)

    # Normalize
    normalized = resized.astype(np.float32) / 255.0

    # Add batch dimension
    if len(normalized.shape) == 2:
        normalized = np.expand_dims(normalized, axis=-1)
        normalized = np.repeat(normalized, 3, axis=-1)

    return np.expand_dims(normalized, axis=0)


def preprocess_microscopy(
    image: np.ndarray, target_size: Tuple[int, int] = (224, 224)
) -> np.ndarray:
    """Preprocess microscopy image for model input"""
    # Resize
    resized = cv2.resize(image, target_size)

    # Normalize
    normalized = resized.astype(np.float32) / 255.0

    # Add batch dimension
    if len(normalized.shape) == 2:
        normalized = np.expand_dims(normalized, axis=-1)
        normalized = np.repeat(normalized, 3, axis=-1)

    return np.expand_dims(normalized, axis=0)


def save_processed_image(image: np.ndarray, output_path: str) -> str:
    """Save processed image"""
    # Convert to uint8
    if image.dtype == np.float32 or image.dtype == np.float64:
        image = (image * 255).astype(np.uint8)

    # Convert RGB to BGR for OpenCV
    if len(image.shape) == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    cv2.imwrite(output_path, image)
    return output_path
