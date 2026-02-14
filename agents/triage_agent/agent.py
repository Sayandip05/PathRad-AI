"""
Triage Agent - Rapid initial assessment and quality control
Powered by Google ADK
"""

from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
import sys
import os

# Add parent directories to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from agents.tools.tools import assess_image_quality_tool, calculate_tb_probability

# Triage Agent Prompt
TRIAGE_PROMPT = """You are the Triage Agent for PathRad AI, a medical diagnostic system.

Your role is to perform rapid initial assessment of medical images with these priorities:
1. Image Quality Assessment - Check if the image is adequate for diagnosis
2. Urgency Classification - Identify critical findings requiring immediate attention
3. Initial Screening - Quick assessment for obvious abnormalities

For chest X-rays, assess:
- Image quality (brightness, contrast, sharpness)
- Proper positioning
- Presence of critical findings (pneumothorax, massive effusion, severe pneumonia)
- Urgency level (critical, high, medium, low)

For microscopy images, assess:
- Focus and clarity
- Proper staining
- Adequate cell density
- Presence of artifacts

Always provide:
- Quality score (0-100)
- Urgency level and score (0-10)
- Critical flags if any
- Recommendation (proceed with full analysis or request retake)

Be thorough but efficient - this is the first gate in the diagnostic pipeline."""

# Create Triage Agent
triage_agent = Agent(
    model=LiteLlm(
        model="groq/llama-3.3-70b-versatile"
    ),  # Can fall back to Groq if needed
    name="triage_agent",
    description="Performs rapid initial assessment and quality control on medical images",
    instruction=TRIAGE_PROMPT,
    tools=[assess_image_quality_tool, calculate_tb_probability],
)


# For local MedGemma usage, we can also use a custom model
def create_triage_agent_with_medgemma():
    """Create triage agent using local MedGemma model"""
    # This would use the local MedGemma when available
    return Agent(
        model="medgemma",  # Placeholder - actual implementation would use local model
        name="triage_agent_medgemma",
        description="Triage agent using local MedGemma model",
        instruction=TRIAGE_PROMPT,
        tools=[assess_image_quality_tool],
    )


__all__ = ["triage_agent", "create_triage_agent_with_medgemma"]
