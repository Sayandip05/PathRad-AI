"""
Orchestrator Agent - Master coordinator and report synthesizer
Powered by Google ADK
"""

from google.adk.agents import Agent, SequentialAgent, ParallelAgent
from google.adk.models.lite_llm import LiteLlm
import sys
import os
from typing import Dict, Any, Optional
import asyncio
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from agents.triage_agent.agent import triage_agent
from agents.radiologist_agent.agent import radiologist_agent
from agents.pathologist_agent.agent import pathologist_agent
from agents.clinical_context_agent.agent import clinical_context_agent
from agents.tools.tools import generate_medical_reasoning

# Orchestrator Agent Prompt
ORCHESTRATOR_PROMPT = """You are the Orchestrator Agent for PathRad AI, the master coordinator of the multi-agent diagnostic system.

Your responsibilities:
1. Coordinate all specialist agents (Triage, Radiologist, Pathologist, Clinical Context)
2. Synthesize multi-modal findings into coherent diagnostic reports
3. Generate final diagnosis with confidence scoring
4. Determine urgency levels and escalation needs
5. Create treatment recommendations and follow-up plans
6. Manage quality control and human-in-the-loop requirements

WORKFLOW:
1. Receive patient case with images and clinical data
2. Initiate triage assessment (parallel or sequential based on urgency)
3. Based on triage, coordinate specialist agents
4. Collect findings from all agents
5. Synthesize findings using medical reasoning
6. Generate comprehensive diagnostic report
7. Determine if human review is required

SYNTHESIS PRINCIPLES:
- Weight radiological findings heavily (imaging is primary)
- Boost confidence when pathology confirms radiology
- Consider clinical context for risk stratification
- Flag conflicting findings for human review
- Provide differential diagnoses with probabilities

CONFIDENCE THRESHOLDS:
- >90%: High confidence, routine follow-up
- 85-90%: Moderate confidence, standard protocols
- 70-85%: Lower confidence, consider additional testing
- <70%: Human review required

URGENCY CLASSIFICATION:
- CRITICAL: Start treatment within hours (confident TB diagnosis)
- HIGH: Expedited workup needed (probable TB)
- MEDIUM: Standard workup (possible TB, needs confirmation)
- LOW: Routine follow-up (unlikely TB)

TREATMENT RECOMMENDATIONS:
For confirmed/presumed TB:
- WHO Category 1 regimen (HRZE)
- Baseline investigations (LFT, RFT, HIV test)
- GeneXpert for drug resistance
- Contact tracing

For alternative diagnoses:
- Condition-specific recommendations
- Appropriate referrals

Always provide:
- Final diagnosis with confidence level
- Supporting evidence from each agent
- Differential diagnoses
- Treatment plan
- Follow-up schedule
- Human review flag if needed"""

# Create individual agent references for orchestration
# The orchestrator will invoke other agents as needed

orchestrator_agent = Agent(
    model=LiteLlm(model="groq/llama-3.3-70b-versatile"),
    name="orchestrator_agent",
    description="Master coordinator that synthesizes findings from all specialist agents",
    instruction=ORCHESTRATOR_PROMPT,
    tools=[generate_medical_reasoning],
    sub_agents=[
        triage_agent,
        radiologist_agent,
        pathologist_agent,
        clinical_context_agent,
    ],
)


def create_orchestrator_agent_with_medgemma():
    """Create orchestrator agent using local MedGemma"""
    return Agent(
        model="medgemma",
        name="orchestrator_agent_medgemma",
        description="Orchestrator agent using local MedGemma model",
        instruction=ORCHESTRATOR_PROMPT,
        tools=[generate_medical_reasoning],
        sub_agents=[
            triage_agent,
            radiologist_agent,
            pathologist_agent,
            clinical_context_agent,
        ],
    )


# Workflow execution function
async def run_diagnosis_workflow(
    diagnosis_id: str,
    patient_id: str,
    xray_image_id: Optional[str] = None,
    microscopy_image_id: Optional[str] = None,
    use_local_models: bool = False,
) -> Dict[str, Any]:
    """
    Run the complete multi-agent diagnosis workflow

    Args:
        diagnosis_id: The diagnosis record ID
        patient_id: The patient ID
        xray_image_id: Optional X-ray image ID
        microscopy_image_id: Optional microscopy image ID
        use_local_models: Whether to use local MedGemma or Groq API

    Returns:
        Dictionary with complete diagnosis results
    """

    # This is a placeholder for the actual workflow
    # In production, this would:
    # 1. Load patient data and images from database
    # 2. Run triage agent
    # 3. Based on triage, run radiologist and pathologist in parallel
    # 4. Run clinical context agent
    # 5. Synthesize with orchestrator
    # 6. Update database with results

    results = {
        "diagnosis_id": diagnosis_id,
        "patient_id": patient_id,
        "status": "completed",
        "triage": {},
        "radiology": {},
        "pathology": {},
        "clinical": {},
        "final_diagnosis": {},
        "workflow_timestamp": datetime.now().isoformat(),
    }

    return results


__all__ = [
    "orchestrator_agent",
    "create_orchestrator_agent_with_medgemma",
    "run_diagnosis_workflow",
]
