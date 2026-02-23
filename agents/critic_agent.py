"""
agents/critic_agent.py

Critic Agent — Evidence quality auditor + false positive detector.

Uses Llama 3.2 to:
    1. Audit rule consistency
    2. Detect contradictions and reinforcement
    3. Identify false positive patterns
    4. Assess signal strength and evidence sufficiency
    5. Recommend whether Judge should trust/lower/raise the score
"""

import json
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any

from app.schemas.evidence import EvidenceCaseFile
from app.schemas.critic import CriticReport
from app.services.llm_clients import LlamaClient, get_critic_client, LLMFatalError


logger = logging.getLogger("forensic.critic")

_PROMPT_DIR = Path(__file__).resolve().parent.parent / "prompts"
_PROMPT_FILE = _PROMPT_DIR / "critic_prompt.txt"
_FEWSHOT_FILE = _PROMPT_DIR / "fewshot_examples.json"


# ═══════════════════════════════════════════════════════
# PROMPT LOADER
# ═══════════════════════════════════════════════════════

class PromptLoader:
    _template: Optional[str] = None
    _fewshots: Optional[List[Dict[str, Any]]] = None

    @classmethod
    def get_template(cls) -> str:
        if cls._template is None:
            if not _PROMPT_FILE.exists():
                raise FileNotFoundError(
                    f"Critic prompt not found: {_PROMPT_FILE}"
                )
            cls._template = _PROMPT_FILE.read_text(encoding="utf-8")
            logger.info(f"Loaded critic prompt ({len(cls._template)} chars)")
        return cls._template

    @classmethod
    def get_fewshots(cls) -> List[Dict[str, Any]]:
        if cls._fewshots is None:
            if not _FEWSHOT_FILE.exists():
                logger.warning("No few-shot file found — running without examples")
                cls._fewshots = []
                return cls._fewshots
            raw = json.loads(_FEWSHOT_FILE.read_text(encoding="utf-8"))
            if not isinstance(raw, list):
                raise ValueError("Few-shot file must be a list")
            cls._fewshots = raw
            logger.info(f"Loaded {len(cls._fewshots)} few-shot examples")
        return cls._fewshots

    @classmethod
    def clear_cache(cls):
        cls._template = None
        cls._fewshots = None


# ═══════════════════════════════════════════════════════
# FEW-SHOT FORMATTER
# ═══════════════════════════════════════════════════════

def format_fewshot_examples(examples: List[Dict[str, Any]]) -> str:
    blocks = []
    for i, ex in enumerate(examples, 1):
        label = ex.get("label", f"Example {i}")
        input_json = json.dumps(ex["input"], indent=2)
        output_json = json.dumps(ex["expected_output"], indent=2)
        block = (
            f"--- Example {i}: {label} ---\n\n"
            f"Case File:\n{input_json}\n\n"
            f"Correct Critic Output:\n{output_json}\n"
        )
        blocks.append(block)
    return "\n".join(blocks)


# ═══════════════════════════════════════════════════════
# CASE FILE SERIALIZER
# ═══════════════════════════════════════════════════════

def serialize_case_file(case: EvidenceCaseFile) -> str:
    data = case.model_dump(exclude_none=True)
    return json.dumps(data, indent=2, default=str)


# ═══════════════════════════════════════════════════════
# PRE-FLIGHT VALIDATION
# ═══════════════════════════════════════════════════════

def preflight_check(case: EvidenceCaseFile) -> Optional[str]:
    """
    Deterministic sanity checks + false positive pattern hints.
    Feeds warnings into the LLM prompt as additional context.
    """
    warnings = []

    # Score math check
    raw_sum = case.metadata.score + case.font.score + case.compression.score
    expected_score = min(raw_sum, 10)
    if case.deterministic_score != expected_score:
        warnings.append(
            f"Score mismatch: modules sum to {raw_sum}, "
            f"expected min({raw_sum},10)={expected_score}, "
            f"but deterministic_score={case.deterministic_score}"
        )

    # High score, no signals (likely false positive)
    for name, module in [
        ("metadata", case.metadata),
        ("font", case.font),
        ("compression", case.compression),
    ]:
        if module.score >= 5 and len(module.signals) == 0:
            warnings.append(
                f"⚠ FP RISK: {name} scores {module.score} but has no signals"
            )

    # Zero score, many signals (possible undercount)
    for name, module in [
        ("metadata", case.metadata),
        ("font", case.font),
        ("compression", case.compression),
    ]:
        if module.score == 0 and len(module.signals) >= 3:
            warnings.append(
                f"{name} scores 0 but has {len(module.signals)} signals"
            )

    # Single module dominance (common FP pattern)
    scores = [case.metadata.score, case.font.score, case.compression.score]
    non_zero = [s for s in scores if s > 0]
    if len(non_zero) == 1 and case.deterministic_score >= 5:
        warnings.append(
            f"⚠ FP RISK: Only one module contributes to score {case.deterministic_score} — "
            f"single-source evidence is often a false positive"
        )

    # Scanner workflow pattern
    scanner_signals = {"multiple_tools_detected", "single_tool_detected"}
    meta_signals = set(case.metadata.signals)
    if meta_signals & scanner_signals and case.compression.score <= 2:
        if case.font.score <= 2:
            warnings.append(
                "⚠ FP RISK: Metadata tool signals + clean font/compression "
                "is consistent with normal scanner workflow"
            )

    if warnings:
        return "; ".join(warnings)
    return None


# ═══════════════════════════════════════════════════════
# CRITIC AGENT
# ═══════════════════════════════════════════════════════

class CriticAgent:
    """
    Evidence quality auditor + false positive detector.

    Receives: EvidenceCaseFile
    Returns: CriticReport (consistency + FP assessment)

    Does NOT make the tampered/not-tampered call.
    That is the Judge's job.
    """

    def __init__(self, client: Optional[LlamaClient] = None):
        self._client = client

    @property
    def client(self) -> LlamaClient:
        if self._client is None:
            self._client = get_critic_client()
        return self._client

    def _build_system_prompt(self) -> str:
        template = PromptLoader.get_template()
        examples = PromptLoader.get_fewshots()
        if examples:
            formatted = format_fewshot_examples(examples)
        else:
            formatted = "(No examples provided — rely on instructions above.)"
        return template.replace("{few_shot_examples}", formatted)

    def _build_user_prompt(
        self,
        case: EvidenceCaseFile,
        preflight_warning: Optional[str] = None,
    ) -> str:
        case_json = serialize_case_file(case)
        parts = [
            "Analyze the following case file.",
            "Assess consistency AND false positive risk.",
            "Produce your audit report.",
            "",
            "Case File:",
            case_json,
        ]
        if preflight_warning:
            parts.extend([
                "",
                "⚠ PREFLIGHT WARNINGS (automated detection before your analysis):",
                preflight_warning,
                "",
                "Consider these patterns carefully in your FP assessment.",
            ])
        return "\n".join(parts)

    async def audit(self, case: EvidenceCaseFile) -> CriticReport:
        logger.info(f"[{case.case_id}] Starting critic audit + FP assessment")

        preflight_warning = preflight_check(case)
        if preflight_warning:
            logger.warning(f"[{case.case_id}] Preflight: {preflight_warning}")

        system_prompt = self._build_system_prompt()
        user_prompt = self._build_user_prompt(case, preflight_warning)

        logger.debug(
            f"[{case.case_id}] System: {len(system_prompt)} chars | "
            f"User: {len(user_prompt)} chars"
        )

        try:
            report: CriticReport = await self.client.generate(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                schema=CriticReport,
            )
        except LLMFatalError:
            logger.error(f"[{case.case_id}] Critic audit FAILED")
            raise

        logger.info(
            f"[{case.case_id}] Critic complete | "
            f"consistent={report.rule_consistency} | "
            f"contradictions={len(report.contradictions)} | "
            f"FP_indicators={len(report.false_positive_indicators)} | "
            f"signal_strength={report.signal_strength.value} | "
            f"sufficiency={report.evidence_sufficiency.value} | "
            f"adjustment={report.recommended_adjustment.value} | "
            f"confidence={report.confidence:.2f}"
        )

        return report
