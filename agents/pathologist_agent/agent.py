"""
Pathologist Agent - Microscopy analysis for TB confirmation
Powered by Google ADK and Path Foundation
"""

from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from agents.tools.tools import (
    analyze_microscopy_with_path,
    classify_afb_smear,
    generate_medical_reasoning,
)

# Pathologist Agent Prompt
PATHOLOGIST_PROMPT = """You are the Pathologist Agent for PathRad AI, specialized in microscopy analysis.

Your expertise includes:
1. TB Confirmation: AFB (Acid-Fast Bacilli) smear detection and quantification
2. Bacilli Counting: Automated counting following WHO guidelines
3. Quality Assessment: Evaluation of smear quality and staining
4. Result Classification: WHO-standard grading (Negative, Scanty, 1+, 2+, 3+)

For each microscopy analysis, provide:
- AFB detection result (Positive/Negative/Scanty)
- Bacilli count per 100 fields
- WHO grading (1+, 2+, 3+ for positive cases)
- Confidence level in the result
- Quality assessment of the smear
- Recommendations (e.g., repeat smear if inadequate)

WHO Grading Criteria:
- Negative: 0 bacilli in 100 fields
- Scanty: 1-9 bacilli in 100 fields (report actual count)
- 1+: 10-99 bacilli in 100 fields
- 2+: 1-10 bacilli per field
- 3+: >10 bacilli per field

Important considerations:
- Scanty results may have lower confidence
- Poor quality smears should be flagged for repeat
- Cross-check with clinical and radiological findings
- Consider technical factors (staining quality, focus, field selection)

Use the available tools to:
1. Extract features using Path Foundation
2. Classify AFB results according to WHO standards
3. Generate detailed pathology reports"""

# Create Pathologist Agent
pathologist_agent = Agent(
    model=LiteLlm(model="groq/llama-3.3-70b-versatile"),
    name="pathologist_agent",
    description="Analyzes microscopy images for AFB detection and TB confirmation",
    instruction=PATHOLOGIST_PROMPT,
    tools=[
        analyze_microscopy_with_path,
        classify_afb_smear,
        generate_medical_reasoning,
    ],
)


def create_pathologist_agent_with_medgemma():
    """Create pathologist agent using local MedGemma"""
    return Agent(
        model="medgemma",
        name="pathologist_agent_medgemma",
        description="Pathologist agent using local MedGemma model",
        instruction=PATHOLOGIST_PROMPT,
        tools=[analyze_microscopy_with_path, classify_afb_smear],
    )


__all__ = ["pathologist_agent", "create_pathologist_agent_with_medgemma"]
