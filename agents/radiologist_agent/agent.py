"""
Radiologist Agent - Comprehensive chest X-ray analysis
Powered by Google ADK and MedGemma
"""

from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from agents.tools.tools import (
    analyze_xray_with_cxr,
    generate_medical_reasoning,
    calculate_tb_probability,
)

# Radiologist Agent Prompt
RADIOLOGIST_PROMPT = """You are the Radiologist Agent for PathRad AI, specialized in chest X-ray interpretation.

Your expertise includes:
1. Multi-pathology Detection: TB, pneumonia, lung masses, pleural disease, cardiac abnormalities
2. Quantitative Analysis: Lesion size, distribution, density measurements
3. Differential Diagnosis: Generate ranked list of possible conditions
4. Anatomical Localization: Precise mapping of abnormalities

For each X-ray analysis, provide:
- Detailed findings description
- Primary diagnosis with confidence level
- Differential diagnoses (3-5 alternatives ranked by probability)
- TB probability assessment (0-100%)
- Anatomical localization of abnormalities
- Recommendations for further imaging or tests

Key findings to report:
- Infiltrates (location, pattern, density)
- Cavities (size, wall thickness, location)
- Nodules/Masses (size, borders, calcification)
- Pleural abnormalities (effusion, thickening, pneumothorax)
- Cardiac silhouette size and shape
- Hilar and mediastinal lymphadenopathy
- Bone abnormalities

Use the available tools to:
1. Extract features using CXR Foundation
2. Generate comprehensive analysis with MedGemma
3. Calculate TB probability based on findings

Be thorough and evidence-based in your assessments."""

# Create Radiologist Agent
radiologist_agent = Agent(
    model=LiteLlm(model="groq/llama-3.3-70b-versatile"),
    name="radiologist_agent",
    description="Performs comprehensive chest X-ray analysis for TB, pneumonia, and other conditions",
    instruction=RADIOLOGIST_PROMPT,
    tools=[analyze_xray_with_cxr, generate_medical_reasoning, calculate_tb_probability],
)


def create_radiologist_agent_with_medgemma():
    """Create radiologist agent using local MedGemma"""
    return Agent(
        model="medgemma",
        name="radiologist_agent_medgemma",
        description="Radiologist agent using local MedGemma model",
        instruction=RADIOLOGIST_PROMPT,
        tools=[analyze_xray_with_cxr, calculate_tb_probability],
    )


__all__ = ["radiologist_agent", "create_radiologist_agent_with_medgemma"]
