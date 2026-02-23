"""
schemas/reflection.py

Forensic verdict schema — the Judge's output.

KEY CHANGE: The Judge CAN override rule-based scores.
Rules are features/evidence, not authority.
"""

from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum


class Severity(str, Enum):
    Low = "Low"
    Medium = "Medium"
    High = "High"


class EvidenceSource(str, Enum):
    METADATA = "metadata"
    FONT = "font"
    COMPRESSION = "compression"
    QR = "qr"
    RULE = "rule"
    CONTRADICTION = "contradiction"
    FALSE_POSITIVE = "false_positive"


class EvidenceWeight(str, Enum):
    STRONG_SUPPORTING = "strong_supporting"
    SUPPORTING = "supporting"
    WEAK_SUPPORTING = "weak_supporting"
    NEUTRAL = "neutral"
    CONTRADICTING = "contradicting"
    FALSE_POSITIVE_PATTERN = "false_positive_pattern"


class EvidenceItem(BaseModel):
    source: EvidenceSource
    finding: str = Field(..., max_length=200)
    weight: EvidenceWeight

    class Config:
        extra = "forbid"


class ConfidenceLevel(str, Enum):
    VERY_LOW = "very_low"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


class ForensicVerdict(BaseModel):
    """
    Final verdict from the Learned Judge.

    The Judge receives rule scores as EVIDENCE and can
    OVERRIDE them based on reasoning.

    rule_based_* fields = what rules said (read-only reference)
    tampered/severity/probability = what the Judge decided
    """

    case_id: str

    # ── What the rules said (reference only, never modified) ──
    rule_based_score: int = Field(
        ..., ge=0, le=10,
        description="Original deterministic score (preserved for audit trail)"
    )
    rule_based_priority: str = Field(
        ...,
        description="Original rule priority (preserved for audit trail)"
    )

    # ── What the Judge decided (THIS is the actual output) ──
    tampered: bool = Field(
        ..., description="Judge's tampering decision"
    )
    tampered_probability: float = Field(
        ..., ge=0, le=1,
        description="Judge's assessed probability of tampering (0.0-1.0)"
    )
    severity: Severity = Field(
        ..., description="Judge-determined severity (CAN differ from rules)"
    )

    # ── Override tracking ──
    override_applied: bool = Field(
        ...,
        description="Did the Judge disagree with the rule-based assessment?"
    )
    override_reason: Optional[str] = Field(
        default=None, max_length=300,
        description="Why the Judge overrode the rules (required if override_applied=true)"
    )

    # ── Explanation + Evidence ──
    explanation: str = Field(..., max_length=500)
    evidence: List[EvidenceItem] = Field(..., min_length=1, max_length=20)

    # ── Confidence ──
    confidence: float = Field(..., ge=0, le=1)
    confidence_level: ConfidenceLevel

    # ── Review flag ──
    flagged_for_human_review: bool = False
    review_reason: Optional[str] = None

    class Config:
        extra = "forbid"
