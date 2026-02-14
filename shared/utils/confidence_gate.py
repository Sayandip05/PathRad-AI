"""
Confidence Threshold System and Human Review Escalation Logic
Clinical-grade decision making for diagnostic confidence
"""

from typing import Dict, Any, List, Tuple
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ConfidenceLevel(Enum):
    """Confidence level categories"""

    HIGH = "high"  # ≥ 0.85
    MEDIUM = "medium"  # 0.75 - 0.85
    LOW = "low"  # < 0.75


class EscalationReason(Enum):
    """Reasons for escalating to human review"""

    LOW_CONFIDENCE = "low_confidence"
    AGENT_DISAGREEMENT = "agent_disagreement"
    IMAGE_QUALITY_LOW = "image_quality_low"
    MODEL_ERROR = "model_error"
    CLINICAL_CONTRADICTION = "clinical_contradiction"
    HIGH_RISK_FINDING = "high_risk_finding"


# Confidence thresholds
CONFIDENCE_THRESHOLDS = {
    "auto_approve": 0.85,  # ≥ 0.85: Auto-approve
    "warning": 0.75,  # 0.75 - 0.85: Flag warning
    "escalate": 0.75,  # < 0.75: Escalate to human
}


class ConfidenceGate:
    """
    Clinical-grade confidence assessment and escalation logic
    """

    @staticmethod
    def assess_confidence_level(confidence_score: float) -> ConfidenceLevel:
        """
        Determine confidence level based on score
        """
        if confidence_score >= CONFIDENCE_THRESHOLDS["auto_approve"]:
            return ConfidenceLevel.HIGH
        elif confidence_score >= CONFIDENCE_THRESHOLDS["escalate"]:
            return ConfidenceLevel.MEDIUM
        else:
            return ConfidenceLevel.LOW

    @staticmethod
    def check_agent_agreement(
        agent_results: List[Dict[str, Any]],
    ) -> Tuple[bool, float, str]:
        """
        Check if agents agree on diagnosis
        Returns: (agreement_ok, disagreement_score, reason)
        """
        if len(agent_results) < 2:
            return True, 0.0, "Single agent result"

        # Extract diagnoses
        diagnoses = []
        for result in agent_results:
            if "primary_diagnosis" in result:
                diagnoses.append(result["primary_diagnosis"].lower())

        if len(diagnoses) < 2:
            return True, 0.0, "Insufficient diagnoses to compare"

        # Check if all agents agree
        unique_diagnoses = set(diagnoses)
        agreement_ratio = 1.0 / len(unique_diagnoses)

        if len(unique_diagnoses) == 1:
            return True, 1.0, "All agents agree"
        elif agreement_ratio >= 0.5:
            return (
                True,
                agreement_ratio,
                f"Partial agreement: {len(unique_diagnoses)} different diagnoses",
            )
        else:
            return False, agreement_ratio, f"Agents disagree: {unique_diagnoses}"

    @classmethod
    def evaluate_final_diagnosis(
        cls,
        confidence_score: float,
        agent_results: List[Dict[str, Any]],
        image_quality_score: float,
        clinical_risk_factors: List[str],
        is_tb_suspected: bool,
    ) -> Dict[str, Any]:
        """
        Comprehensive evaluation for final diagnosis
        Returns complete assessment with escalation decision
        """
        logger.info(
            f"Evaluating diagnosis - Confidence: {confidence_score}, Quality: {image_quality_score}"
        )

        assessment = {
            "confidence_score": confidence_score,
            "confidence_level": cls.assess_confidence_level(confidence_score).value,
            "requires_human_review": False,
            "escalation_reasons": [],
            "risk_factors": [],
            "recommendation": "",
            "auto_approve": False,
        }

        # Check 1: Confidence threshold
        if confidence_score < CONFIDENCE_THRESHOLDS["escalate"]:
            assessment["requires_human_review"] = True
            assessment["escalation_reasons"].append(
                {
                    "reason": EscalationReason.LOW_CONFIDENCE.value,
                    "details": f"Confidence {confidence_score:.2f} below threshold {CONFIDENCE_THRESHOLDS['escalate']}",
                }
            )
            logger.warning(f"Low confidence detected: {confidence_score}")

        # Check 2: Agent agreement
        agreement_ok, agreement_score, agreement_reason = cls.check_agent_agreement(
            agent_results
        )
        if not agreement_ok:
            assessment["requires_human_review"] = True
            assessment["escalation_reasons"].append(
                {
                    "reason": EscalationReason.AGENT_DISAGREEMENT.value,
                    "details": agreement_reason,
                    "agreement_score": agreement_score,
                }
            )
            logger.warning(f"Agent disagreement: {agreement_reason}")

        # Check 3: Image quality
        if image_quality_score < 60:
            assessment["requires_human_review"] = True
            assessment["escalation_reasons"].append(
                {
                    "reason": EscalationReason.IMAGE_QUALITY_LOW.value,
                    "details": f"Image quality score {image_quality_score}/100 insufficient",
                }
            )
            logger.warning(f"Low image quality: {image_quality_score}")

        # Check 4: Clinical risk factors (auto-escalate high-risk cases)
        high_risk_factors = [
            "hiv",
            "immunocompromised",
            "severe_malnutrition",
            "miliary_tb",
        ]
        for factor in clinical_risk_factors:
            if any(risk in factor.lower() for risk in high_risk_factors):
                assessment["risk_factors"].append(factor)
                # Don't auto-escalate, but flag as high risk
                logger.info(f"High risk factor detected: {factor}")

        # Check 5: TB suspected with high severity
        if is_tb_suspected and confidence_score >= 0.70:
            # High confidence TB cases need careful review
            assessment["risk_factors"].append("High-confidence TB detection")
            if confidence_score < 0.80:
                assessment["requires_human_review"] = True
                assessment["escalation_reasons"].append(
                    {
                        "reason": EscalationReason.HIGH_RISK_FINDING.value,
                        "details": "TB suspected with moderate confidence - requires confirmation",
                    }
                )

        # Determine final recommendation
        if assessment["requires_human_review"]:
            assessment["recommendation"] = "ESCALATE_TO_HUMAN_REVIEW"
            assessment["auto_approve"] = False
        elif confidence_score >= CONFIDENCE_THRESHOLDS["auto_approve"]:
            assessment["recommendation"] = "AUTO_APPROVE"
            assessment["auto_approve"] = True
        else:
            assessment["recommendation"] = "STANDARD_REVIEW"
            assessment["auto_approve"] = False

        logger.info(
            f"Assessment complete: {assessment['recommendation']}, Escalate: {assessment['requires_human_review']}"
        )

        return assessment

    @staticmethod
    def calculate_composite_confidence(
        agent_confidences: Dict[str, float], weights: Dict[str, float] = None
    ) -> float:
        """
        Calculate weighted composite confidence from multiple agents
        """
        if not agent_confidences:
            return 0.0

        # Default weights
        if weights is None:
            weights = {
                "radiologist": 0.40,
                "pathologist": 0.30,
                "clinical_context": 0.20,
                "triage": 0.10,
            }

        # Calculate weighted average
        total_weight = 0
        weighted_sum = 0

        for agent, confidence in agent_confidences.items():
            weight = weights.get(agent, 0.1)
            weighted_sum += confidence * weight
            total_weight += weight

        if total_weight == 0:
            return 0.0

        composite = weighted_sum / total_weight

        # Apply penalty for missing agents
        expected_agents = set(weights.keys())
        present_agents = set(agent_confidences.keys())
        missing_agents = expected_agents - present_agents

        if missing_agents:
            penalty = len(missing_agents) * 0.05  # 5% penalty per missing agent
            composite = max(0.0, composite - penalty)
            logger.warning(
                f"Missing agents: {missing_agents}, applying penalty: {penalty}"
            )

        return round(composite, 3)


# Convenience functions
def evaluate_diagnosis(
    confidence: float,
    agent_results: List[Dict[str, Any]],
    image_quality: float = 100.0,
    risk_factors: List[str] = None,
    is_tb_suspected: bool = False,
) -> Dict[str, Any]:
    """Main entry point for diagnosis evaluation"""
    gate = ConfidenceGate()
    return gate.evaluate_final_diagnosis(
        confidence_score=confidence,
        agent_results=agent_results,
        image_quality_score=image_quality,
        clinical_risk_factors=risk_factors or [],
        is_tb_suspected=is_tb_suspected,
    )


def should_escalate(assessment: Dict[str, Any]) -> bool:
    """Quick check if case should be escalated"""
    return assessment.get("requires_human_review", False)
