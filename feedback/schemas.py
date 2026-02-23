"""
feedback/schemas.py

Data models for human feedback and case outcomes.
"""

from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum
from datetime import datetime


class HumanLabel(str, Enum):
    TAMPERED = "tampered"
    NOT_TAMPERED = "not_tampered"
    UNCERTAIN = "uncertain"


class FeedbackSubmission(BaseModel):
    """What a human reviewer submits."""
    case_id: str
    human_label: HumanLabel
    human_notes: Optional[str] = Field(default=None, max_length=500)
    reviewed_by: Optional[str] = None

    class Config:
        extra = "forbid"


class StoredCase(BaseModel):
    """Complete case record stored in feedback DB."""
    case_id: str

    # Input snapshot
    case_data: dict

    # Agent outputs
    critic_report: dict
    verdict: dict

    # Judge assessment
    judge_tampered: bool
    judge_probability: float
    judge_severity: str
    override_applied: bool

    # Rule-based assessment
    rule_score: int
    rule_priority: str

    # Human feedback (null until reviewed)
    human_label: Optional[str] = None
    human_notes: Optional[str] = None
    reviewed_by: Optional[str] = None
    reviewed_at: Optional[str] = None

    # Metadata
    created_at: Optional[str] = None
    pipeline_duration_ms: Optional[float] = None

    # Learning status
    embedded: bool = False


class LabeledExample(BaseModel):
    """A labeled case ready for injection into prompts."""
    case_id: str
    case_data: dict
    judge_tampered: bool
    judge_probability: float
    judge_severity: str
    override_applied: bool
    rule_score: int
    rule_priority: str
    human_label: str
    human_notes: Optional[str] = None
    similarity_score: Optional[float] = None

    def was_correct(self) -> bool:
        """Did the Judge agree with the human?"""
        if self.human_label == "tampered":
            return self.judge_tampered is True
        elif self.human_label == "not_tampered":
            return self.judge_tampered is False
        return False

    def to_prompt_text(self) -> str:
        """Format for injection into LLM prompt."""
        correct = "CORRECT" if self.was_correct() else "INCORRECT"
        override_text = ""
        if self.override_applied:
            override_text = (
                f"\n  Override: Rules said {self.rule_priority} "
                f"but Judge said {self.judge_severity}"
            )

        return (
            f"--- Past Case: {self.case_id} ({correct}) ---\n"
            f"  Rule score: {self.rule_score} ({self.rule_priority})\n"
            f"  Judge: tampered={self.judge_tampered}, "
            f"probability={self.judge_probability:.2f}, "
            f"severity={self.judge_severity}"
            f"{override_text}\n"
            f"  Human label: {self.human_label}\n"
            f"  Signals: {self._summarize_signals()}\n"
        )

    def _summarize_signals(self) -> str:
        parts = []
        for module in ["metadata", "font", "compression"]:
            mod_data = self.case_data.get(module, {})
            signals = mod_data.get("signals", [])
            score = mod_data.get("score", 0)
            if signals:
                parts.append(f"{module}({score}): {', '.join(signals[:3])}")
            else:
                parts.append(f"{module}({score}): clean")
        return " | ".join(parts)
