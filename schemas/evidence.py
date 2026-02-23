"""
schemas/evidence.py

Input case file from deterministic detection modules.
Rules produce this. LLM receives this as FEATURES, not authority.
"""

from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Dict, Any
from enum import Enum


class ModuleResult(BaseModel):
    score: int = Field(..., ge=0, le=10)
    signals: List[str] = Field(default_factory=list, max_length=50)
    raw: Optional[Dict[str, Any]] = Field(default=None)

    class Config:
        extra = "forbid"


class Priority(str, Enum):
    Low = "Low"
    Medium = "Medium"
    High = "High"


class EvidenceCaseFile(BaseModel):
    case_id: str
    metadata: ModuleResult
    font: ModuleResult
    compression: ModuleResult
    qr: Optional[ModuleResult] = None
    deterministic_score: int = Field(..., ge=0, le=10)
    priority: Priority

    @field_validator('priority')
    @classmethod
    def validate_priority_score_alignment(cls, v, info):
        score = info.data.get('deterministic_score')
        if score is not None:
            if score >= 8 and v != Priority.High:
                raise ValueError(f"Score {score} must be High priority")
            if 5 <= score < 8 and v != Priority.Medium:
                raise ValueError(f"Score {score} must be Medium priority")
            if score < 5 and v != Priority.Low:
                raise ValueError(f"Score {score} must be Low priority")
        return v

    class Config:
        extra = "forbid"
