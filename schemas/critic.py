"""
schemas/critic.py

Critic output schema — now includes false positive assessment.

The Critic audits consistency AND evaluates whether
the rule-based scores are trustworthy or inflated.
"""

from pydantic import BaseModel, Field, field_validator
from typing import List, Optional
from enum import Enum


class ContradictionImpact(str, Enum):
    Low = "Low"
    Medium = "Medium"
    High = "High"


class ContradictionType(str, Enum):
    MODULE_CONFLICT = "module_conflict"
    RULE_BREACH = "rule_breach"
    SIGNAL_MISMATCH = "signal_mismatch"


class Contradiction(BaseModel):
    contradiction_type: ContradictionType
    modules_involved: List[str] = Field(..., min_length=1)
    impact: ContradictionImpact
    description: str = Field(..., max_length=300)
    resolvable: bool = False
    suggested_action: Optional[str] = None

    class Config:
        extra = "forbid"


class ConfidenceLevel(str, Enum):
    VERY_LOW = "very_low"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


class SignalStrength(str, Enum):
    """How strong are the actual signals (independent of scores)?"""
    VERY_WEAK = "very_weak"
    WEAK = "weak"
    MODERATE = "moderate"
    STRONG = "strong"
    VERY_STRONG = "very_strong"


class EvidenceSufficiency(str, Enum):
    """Is there enough cross-module evidence to make a call?"""
    INSUFFICIENT = "insufficient"
    BORDERLINE = "borderline"
    SUFFICIENT = "sufficient"
    OVERWHELMING = "overwhelming"


class AdjustmentDirection(str, Enum):
    """Should the rule score be trusted, lowered, or raised?"""
    LOWER = "lower"
    MAINTAIN = "maintain"
    HIGHER = "higher"


class CriticReport(BaseModel):
    """
    Critic audit report.

    Now includes FALSE POSITIVE assessment — the key addition
    that enables the Judge to override rule-based scores.
    """

    # ── Consistency audit (same as before) ──
    rule_consistency: bool = Field(
        ..., description="Are deterministic rules applied correctly?"
    )
    contradictions: List[Contradiction] = Field(
        default_factory=list, max_length=10
    )
    reinforcement: List[str] = Field(
        default_factory=list, max_length=10
    )

    # ── Confidence (same as before) ──
    confidence: float = Field(..., ge=0, le=1)
    confidence_level: ConfidenceLevel
    confidence_reason: str = Field(..., max_length=300)

    # ── NEW: False positive assessment ──
    false_positive_indicators: List[str] = Field(
        default_factory=list,
        max_length=10,
        description="Patterns suggesting scores are inflated (e.g., 'scanner workflow', 'PDF/A artifacts')"
    )
    signal_strength: SignalStrength = Field(
        ...,
        description="How strong are the actual signals, independent of scores?"
    )
    evidence_sufficiency: EvidenceSufficiency = Field(
        ...,
        description="Is there enough cross-module evidence?"
    )
    recommended_adjustment: AdjustmentDirection = Field(
        ...,
        description="Should the Judge trust, lower, or raise the rule score?"
    )
    adjustment_reason: Optional[str] = Field(
        default=None,
        max_length=300,
        description="Why adjust? (required if not 'maintain')"
    )

    # ── Notes ──
    audit_notes: str = Field(..., max_length=500)
    rerun_recommended: bool = False

    class Config:
        extra = "forbid"

    @field_validator('confidence_level')
    @classmethod
    def validate_confidence_alignment(cls, v, info):
        conf = info.data.get('confidence')
        if conf is not None:
            if conf >= 0.9 and v != ConfidenceLevel.VERY_HIGH:
                raise ValueError("Confidence score-level mismatch")
            if 0.7 <= conf < 0.9 and v != ConfidenceLevel.HIGH:
                raise ValueError("Confidence score-level mismatch")
            if 0.4 <= conf < 0.7 and v != ConfidenceLevel.MEDIUM:
                raise ValueError("Confidence score-level mismatch")
            if 0.2 <= conf < 0.4 and v != ConfidenceLevel.LOW:
                raise ValueError("Confidence score-level mismatch")
            if conf < 0.2 and v != ConfidenceLevel.VERY_LOW:
                raise ValueError("Confidence score-level mismatch")
        return v
