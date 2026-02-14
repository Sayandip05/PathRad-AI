"""
Production-Grade Image Preprocessing Pipeline
Quality gates, automatic preprocessing, GPU-ready tensors
"""

import cv2
import numpy as np
import torch
from typing import Dict, Any, Tuple, Optional
from pathlib import Path
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Quality thresholds
QUALITY_THRESHOLDS = {
    "min_resolution": (512, 512),
    "max_resolution": (4096, 4096),
    "brightness_min": 30,  # 0-255 scale
    "brightness_max": 225,
    "contrast_min": 40,
    "sharpness_min": 100,
    "snr_min": 2.0,
}

# Model input sizes
MODEL_INPUT_SIZES = {"medgemma": (224, 224), "cxr": (512, 512), "path": (224, 224)}


class ImageQualityError(Exception):
    """Raised when image quality is insufficient"""

    pass


class PreprocessingPipeline:
    """Complete image preprocessing with quality gates"""

    @staticmethod
    def validate_image_file(image_path: str) -> bool:
        """Check if file exists and is readable"""
        path = Path(image_path)
        if not path.exists():
            raise ImageQualityError(f"Image file not found: {image_path}")
        if not path.is_file():
            raise ImageQualityError(f"Path is not a file: {image_path}")

        # Try to read
        img = cv2.imread(str(path))
        if img is None:
            raise ImageQualityError(
                f"Invalid image format or corrupted file: {image_path}"
            )

        return True

    @staticmethod
    def check_resolution(image: np.ndarray) -> Tuple[bool, str]:
        """Check image resolution requirements"""
        h, w = image.shape[:2]
        min_h, min_w = QUALITY_THRESHOLDS["min_resolution"]
        max_h, max_w = QUALITY_THRESHOLDS["max_resolution"]

        if h < min_h or w < min_w:
            return False, f"Resolution too low: {w}x{h} (min: {min_w}x{min_h})"
        if h > max_h or w > max_w:
            return False, f"Resolution too high: {w}x{h} (max: {max_w}x{max_h})"

        return True, f"Resolution OK: {w}x{h}"

    @staticmethod
    def calculate_brightness(image: np.ndarray) -> float:
        """Calculate mean brightness"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        return float(np.mean(gray))

    @staticmethod
    def calculate_contrast(image: np.ndarray) -> float:
        """Calculate standard deviation (contrast)"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        return float(np.std(gray))

    @staticmethod
    def calculate_sharpness(image: np.ndarray) -> float:
        """Calculate Laplacian variance (sharpness)"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        return float(laplacian.var())

    @staticmethod
    def calculate_snr(image: np.ndarray) -> float:
        """Calculate signal-to-noise ratio"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        mean = np.mean(gray)
        std = np.std(gray) + 1e-10
        return float(mean / std)

    @classmethod
    def full_quality_assessment(cls, image_path: str) -> Dict[str, Any]:
        """
        Complete quality assessment with all checks
        Returns detailed report and pass/fail status
        """
        logger.info(f"Starting quality assessment for: {image_path}")

        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ImageQualityError(f"Failed to load image: {image_path}")

        # Run all quality checks
        checks = {}

        # Resolution check
        res_ok, res_msg = cls.check_resolution(image)
        checks["resolution"] = {
            "passed": res_ok,
            "message": res_msg,
            "width": image.shape[1],
            "height": image.shape[0],
        }

        # Brightness check
        brightness = cls.calculate_brightness(image)
        brightness_ok = (
            QUALITY_THRESHOLDS["brightness_min"]
            < brightness
            < QUALITY_THRESHOLDS["brightness_max"]
        )
        checks["brightness"] = {
            "passed": brightness_ok,
            "value": brightness,
            "min": QUALITY_THRESHOLDS["brightness_min"],
            "max": QUALITY_THRESHOLDS["brightness_max"],
        }

        # Contrast check
        contrast = cls.calculate_contrast(image)
        contrast_ok = contrast > QUALITY_THRESHOLDS["contrast_min"]
        checks["contrast"] = {
            "passed": contrast_ok,
            "value": contrast,
            "min": QUALITY_THRESHOLDS["contrast_min"],
        }

        # Sharpness check
        sharpness = cls.calculate_sharpness(image)
        sharpness_ok = sharpness > QUALITY_THRESHOLDS["sharpness_min"]
        checks["sharpness"] = {
            "passed": sharpness_ok,
            "value": sharpness,
            "min": QUALITY_THRESHOLDS["sharpness_min"],
        }

        # SNR check
        snr = cls.calculate_snr(image)
        snr_ok = snr > QUALITY_THRESHOLDS["snr_min"]
        checks["snr"] = {
            "passed": snr_ok,
            "value": snr,
            "min": QUALITY_THRESHOLDS["snr_min"],
        }

        # Overall assessment
        all_passed = all(check["passed"] for check in checks.values())
        failed_checks = [name for name, check in checks.items() if not check["passed"]]

        quality_score = sum(20 for check in checks.values() if check["passed"])

        result = {
            "image_path": image_path,
            "timestamp": datetime.utcnow().isoformat(),
            "overall_passed": all_passed,
            "quality_score": quality_score,
            "checks": checks,
            "failed_checks": failed_checks,
            "requires_human_review": not all_passed,
        }

        logger.info(
            f"Quality assessment complete. Score: {quality_score}/100, Passed: {all_passed}"
        )

        return result

    @staticmethod
    def preprocess_for_model(image_path: str, model_type: str) -> torch.Tensor:
        """
        Preprocess image for specific model
        Returns GPU-ready tensor
        """
        assert model_type in MODEL_INPUT_SIZES, f"Unknown model type: {model_type}"

        # Load image
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ImageQualityError(f"Failed to load image: {image_path}")

        # Get target size
        target_size = MODEL_INPUT_SIZES[model_type]

        # Resize
        image = cv2.resize(image, target_size)

        # Normalize to 0-1
        image = image.astype(np.float32) / 255.0

        # Convert to tensor (1, 1, H, W) for single channel
        tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).float()

        # Move to GPU
        if torch.cuda.is_available():
            tensor = tensor.cuda()
            logger.info(f"Tensor moved to GPU: {torch.cuda.get_device_name(0)}")
        else:
            logger.warning(
                "GPU not available, using CPU (performance will be degraded)"
            )

        return tensor

    @staticmethod
    def preprocess_for_medgemma(image_path: str) -> torch.Tensor:
        """Special preprocessing for MedGemma vision model"""
        image = cv2.imread(image_path)
        if image is None:
            raise ImageQualityError(f"Failed to load image: {image_path}")

        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize to 224x224
        image = cv2.resize(image, (224, 224))

        # Normalize to [-1, 1] range (common for vision transformers)
        image = (image.astype(np.float32) / 127.5) - 1.0

        # Convert to tensor (1, 3, H, W)
        tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float()

        # Move to GPU
        if torch.cuda.is_available():
            tensor = tensor.cuda()

        return tensor

    @classmethod
    def process_image(cls, image_path: str, model_type: str) -> Dict[str, Any]:
        """
        Full pipeline: Quality check + Preprocessing
        Returns processed tensor and quality report
        """
        logger.info(f"Processing image for {model_type}: {image_path}")

        # Step 1: Quality assessment
        quality_report = cls.full_quality_assessment(image_path)

        if not quality_report["overall_passed"]:
            logger.error(
                f"Image quality check failed: {quality_report['failed_checks']}"
            )
            raise ImageQualityError(
                f"Image quality insufficient. Failed checks: {quality_report['failed_checks']}"
            )

        # Step 2: Preprocessing
        if model_type == "medgemma":
            tensor = cls.preprocess_for_medgemma(image_path)
        else:
            tensor = cls.preprocess_for_model(image_path, model_type)

        logger.info(
            f"Image preprocessing complete. Tensor shape: {tensor.shape}, Device: {tensor.device}"
        )

        return {
            "tensor": tensor,
            "quality_report": quality_report,
            "model_type": model_type,
        }


# Convenience functions
def validate_and_preprocess(image_path: str, model_type: str) -> Dict[str, Any]:
    """Main entry point for image preprocessing"""
    pipeline = PreprocessingPipeline()
    return pipeline.process_image(image_path, model_type)


def quick_quality_check(image_path: str) -> bool:
    """Quick check without full preprocessing"""
    try:
        pipeline = PreprocessingPipeline()
        report = pipeline.full_quality_assessment(image_path)
        return report["overall_passed"]
    except Exception as e:
        logger.error(f"Quality check failed: {e}")
        return False
