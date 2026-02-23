"""
memory/embedder.py

Converts case files into text representations for embedding.

The text captures the forensic "fingerprint" of a case:
signals, scores, patterns — so similar cases cluster together.
"""

import logging
from typing import Optional

from app.schemas.evidence import EvidenceCaseFile


logger = logging.getLogger("forensic.memory")


def case_to_text(case: EvidenceCaseFile) -> str:
    """
    Convert a case file into a structured text representation
    suitable for embedding.

    The text is designed so that cases with similar PATTERNS
    (not just similar scores) will be close in embedding space.

    Focus on SIGNALS (qualitative) not just SCORES (quantitative)
    because signals are what distinguish real tampering from FP.
    """
    parts = []

    # Module signals (most important for similarity)
    for name, module in [
        ("metadata", case.metadata),
        ("font", case.font),
        ("compression", case.compression),
    ]:
        if module.signals:
            signals_text = ", ".join(module.signals)
            parts.append(f"{name} score={module.score}: {signals_text}")
        else:
            parts.append(f"{name} score={module.score}: clean")

    if case.qr:
        if case.qr.signals:
            parts.append(f"qr score={case.qr.score}: {', '.join(case.qr.signals)}")
        else:
            parts.append(f"qr score={case.qr.score}: clean")

    # Score pattern
    scores = [case.metadata.score, case.font.score, case.compression.score]
    non_zero = sum(1 for s in scores if s > 0)
    parts.append(
        f"pattern: {non_zero}/3 modules active, "
        f"total={case.deterministic_score}, "
        f"priority={case.priority.value}"
    )

    # Signal density
    total_signals = (
        len(case.metadata.signals)
        + len(case.font.signals)
        + len(case.compression.signals)
    )
    parts.append(f"signal_density: {total_signals} total signals")

    return " | ".join(parts)


def case_with_outcome_to_text(
    case_data: dict,
    judge_tampered: bool,
    judge_severity: str,
    human_label: Optional[str] = None,
) -> str:
    """
    Convert a case + outcome into text for embedding.
    Includes the outcome so we can retrieve cases with
    known results.
    """
    parts = []

    for name in ["metadata", "font", "compression"]:
        mod = case_data.get(name, {})
        signals = mod.get("signals", [])
        score = mod.get("score", 0)
        if signals:
            parts.append(f"{name}({score}): {', '.join(signals[:5])}")
        else:
            parts.append(f"{name}({score}): clean")

    parts.append(f"rule_score={case_data.get('deterministic_score', 0)}")
    parts.append(f"judge={judge_severity}")

    if human_label:
        parts.append(f"human={human_label}")

    return " | ".join(parts)
