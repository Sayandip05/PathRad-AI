"""
Google ADK Tools for PathRad AI Agents
These tools provide functionality that agents can use
"""

from google.adk.tools import tool
from typing import Dict, Any, List, Optional
import cv2
import numpy as np
from datetime import datetime
import sys
import os

# Add parent directories to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from models import model_manager
from shared.utils.image_processing import preprocess_xray, preprocess_microscopy


@tool
def assess_image_quality_tool(image_path: str) -> Dict[str, Any]:
    """
    Assess the quality of a medical image

    Args:
        image_path: Path to the image file

    Returns:
        Dictionary with quality metrics
    """
    try:
        image = cv2.imread(image_path)
        if image is None:
            return {"error": "Could not load image", "adequate": False}

        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Calculate metrics
        brightness = float(np.mean(gray))
        contrast = float(np.std(gray))
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        sharpness = float(laplacian.var())
        snr = float(np.mean(gray) / (np.std(gray) + 1e-10))

        # Quality assessment
        quality_score = 0
        issues = []

        if 30 < brightness < 225:
            quality_score += 25
        else:
            issues.append("Brightness out of range")

        if contrast > 40:
            quality_score += 25
        else:
            issues.append("Low contrast")

        if sharpness > 100:
            quality_score += 25
        else:
            issues.append("Image not sharp")

        if snr > 2:
            quality_score += 25

        return {
            "quality_score": quality_score,
            "brightness": brightness,
            "contrast": contrast,
            "sharpness": sharpness,
            "snr": snr,
            "adequate": quality_score >= 60,
            "issues": issues,
            "timestamp": datetime.utcnow().isoformat(),
        }
    except Exception as e:
        return {"error": str(e), "adequate": False}


@tool
def analyze_xray_with_cxr(image_path: str) -> Dict[str, Any]:
    """
    Analyze chest X-ray using CXR Foundation

    Args:
        image_path: Path to the X-ray image

    Returns:
        Dictionary with analysis results
    """
    try:
        # Load and preprocess image
        image = cv2.imread(image_path)
        if image is None:
            return {"error": "Could not load X-ray image"}

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        processed = preprocess_xray(image)

        # Extract features
        features = model_manager.extract_cxr_features(processed)

        return {
            "features_shape": features.shape,
            "analysis_completed": True,
            "timestamp": datetime.utcnow().isoformat(),
        }
    except Exception as e:
        return {"error": str(e), "analysis_completed": False}


@tool
def analyze_microscopy_with_path(image_path: str) -> Dict[str, Any]:
    """
    Analyze microscopy image using Path Foundation

    Args:
        image_path: Path to the microscopy image

    Returns:
        Dictionary with analysis results
    """
    try:
        # Load and preprocess image
        image = cv2.imread(image_path)
        if image is None:
            return {"error": "Could not load microscopy image"}

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        processed = preprocess_microscopy(image)

        # Extract features
        features = model_manager.extract_path_features(processed)

        return {
            "features_shape": features.shape,
            "analysis_completed": True,
            "timestamp": datetime.utcnow().isoformat(),
        }
    except Exception as e:
        return {"error": str(e), "analysis_completed": False}


@tool
def generate_medical_reasoning(prompt: str, max_tokens: int = 400) -> str:
    """
    Generate medical reasoning using MedGemma

    Args:
        prompt: The prompt for medical reasoning
        max_tokens: Maximum tokens to generate

    Returns:
        Generated reasoning text
    """
    try:
        if model_manager.medgemma_model is None:
            return "Error: MedGemma model not available"

        response = model_manager.medgemma_generate(prompt, max_tokens=max_tokens)
        return response
    except Exception as e:
        return f"Error generating reasoning: {str(e)}"


@tool
def calculate_tb_probability(findings: str, symptoms: List[str]) -> Dict[str, Any]:
    """
    Calculate TB probability based on findings and symptoms

    Args:
        findings: Radiology findings text
        symptoms: List of symptoms

    Returns:
        Dictionary with TB probability and risk assessment
    """
    tb_keywords = ["tuberculosis", "tb", "cavity", "apical", "infiltrate", "miliary"]

    findings_lower = findings.lower()
    keyword_count = sum(1 for kw in tb_keywords if kw in findings_lower)

    # Base probability + keyword boost
    probability = 0.45 + (keyword_count * 0.08)
    probability = min(0.95, probability)

    # Adjust based on symptoms
    if "cough" in [s.lower() for s in symptoms]:
        probability += 0.05
    if "weight loss" in [s.lower() for s in symptoms]:
        probability += 0.05
    if "fever" in [s.lower() for s in symptoms]:
        probability += 0.03

    probability = min(0.98, probability)

    risk_level = (
        "high" if probability > 0.7 else "medium" if probability > 0.4 else "low"
    )

    return {
        "tb_probability": probability,
        "risk_level": risk_level,
        "keywords_detected": keyword_count,
        "assessment_timestamp": datetime.utcnow().isoformat(),
    }


@tool
def classify_afb_smear(bacilli_count: int) -> Dict[str, Any]:
    """
    Classify AFB smear result based on WHO guidelines

    Args:
        bacilli_count: Number of bacilli detected

    Returns:
        Dictionary with classification result
    """
    if bacilli_count == 0:
        result = "Negative"
        quantification = "No acid-fast bacilli seen in 100 fields"
    elif bacilli_count < 10:
        result = "Scanty"
        quantification = f"{bacilli_count} bacilli in 100 fields"
    elif bacilli_count < 100:
        result = "1+"
        quantification = "10-99 bacilli per 100 fields"
    elif bacilli_count < 1000:
        result = "2+"
        quantification = "1-10 bacilli per field"
    else:
        result = "3+"
        quantification = ">10 bacilli per field"

    confidence = (
        0.90
        if bacilli_count == 0 or bacilli_count > 50
        else 0.70
        if bacilli_count < 10
        else 0.85
    )

    return {
        "result": result,
        "quantification": quantification,
        "bacilli_count": bacilli_count,
        "confidence": confidence,
        "classification_timestamp": datetime.utcnow().isoformat(),
    }


@tool
def extract_clinical_factors(history: str, chief_complaint: str) -> Dict[str, Any]:
    """
    Extract clinical risk factors from patient history

    Args:
        history: Clinical history text
        chief_complaint: Chief complaint text

    Returns:
        Dictionary with extracted factors
    """
    text = (history + " " + chief_complaint).lower()

    # Symptom extraction
    symptoms = []
    symptom_keywords = {
        "Productive cough": ["cough", "productive"],
        "Fever": ["fever", "febrile"],
        "Weight loss": ["weight loss", "lost weight"],
        "Night sweats": ["night sweat", "sweating"],
        "Hemoptysis": ["hemoptysis", "blood", "coughing blood"],
        "Fatigue": ["fatigue", "tired", "weakness"],
        "Chest pain": ["chest pain", "thoracic pain"],
    }

    for symptom, keywords in symptom_keywords.items():
        if any(kw in text for kw in keywords):
            symptoms.append(symptom)

    # Risk factors
    risk_factors = []
    if any(kw in text for kw in ["hiv", "aids", "immunodeficiency"]):
        risk_factors.append("HIV/AIDS")
    if any(kw in text for kw in ["diabetes", "diabetic"]):
        risk_factors.append("Diabetes")
    if any(kw in text for kw in ["crowded", "prison", "shelter"]):
        risk_factors.append("Crowded living conditions")
    if any(kw in text for kw in ["smoker", "smoking", "tobacco"]):
        risk_factors.append("Tobacco use")

    # Comorbidities
    comorbidities = []
    if any(kw in text for kw in ["copd", "chronic obstructive"]):
        comorbidities.append("COPD")
    if any(kw in text for kw in ["malnutrition", "underweight"]):
        comorbidities.append("Malnutrition")

    # TB Risk Score
    tb_risk = 3.0  # Base risk
    tb_risk += len(symptoms) * 0.5
    tb_risk += len(comorbidities) * 1.0
    tb_risk += len(risk_factors) * 0.5
    tb_risk = min(10.0, tb_risk)

    return {
        "symptoms": symptoms,
        "risk_factors": risk_factors,
        "comorbidities": comorbidities,
        "tb_risk_score": tb_risk,
        "extraction_timestamp": datetime.utcnow().isoformat(),
    }
