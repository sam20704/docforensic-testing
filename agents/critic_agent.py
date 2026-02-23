"""
agents/critic_agent.py

Critic Agent — Evidence quality auditor + false positive detector.

LEARNING UPGRADE: Now injects similar labeled cases as
dynamic few-shots instead of relying on static examples.
"""

import json
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any

from app.schemas.evidence import EvidenceCaseFile
from app.schemas.critic import CriticReport
from app.services.llm_clients import LlamaClient, get_critic_client, LLMFatalError
from app.config import settings


logger = logging.getLogger("forensic.critic")

_PROMPT_DIR = Path(__file__).resolve().parent.parent / "prompts"
_PROMPT_FILE = _PROMPT_DIR / "critic_prompt.txt"
_FEWSHOT_FILE = _PROMPT_DIR / "fewshot_examples.json"


class PromptLoader:
    _template: Optional[str] = None
    _fewshots: Optional[List[Dict[str, Any]]] = None

    @classmethod
    def get_template(cls) -> str:
        if cls._template is None:
            if not _PROMPT_FILE.exists():
                raise FileNotFoundError(f"Critic prompt not found: {_PROMPT_FILE}")
            cls._template = _PROMPT_FILE.read_text(encoding="utf-8")
            logger.info(f"Loaded critic prompt ({len(cls._template)} chars)")
        return cls._template

    @classmethod
    def get_fewshots(cls) -> List[Dict[str, Any]]:
        if cls._fewshots is None:
            if not _FEWSHOT_FILE.exists():
                logger.warning("No few-shot file — running without static examples")
                cls._fewshots = []
                return cls._fewshots
            raw = json.loads(_FEWSHOT_FILE.read_text(encoding="utf-8"))
            if not isinstance(raw, list):
                raise ValueError("Few-shot file must be a list")
            cls._fewshots = raw
            logger.info(f"Loaded {len(cls._fewshots)} static few-shot examples")
        return cls._fewshots

    @classmethod
    def clear_cache(cls):
        cls._template = None
        cls._fewshots = None


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


def serialize_case_file(case: EvidenceCaseFile) -> str:
    data = case.model_dump(exclude_none=True)
    return json.dumps(data, indent=2, default=str)


def preflight_check(case: EvidenceCaseFile) -> Optional[str]:
    warnings = []

    raw_sum = case.metadata.score + case.font.score + case.compression.score
    expected_score = min(raw_sum, 10)
    if case.deterministic_score != expected_score:
        warnings.append(
            f"Score mismatch: sum={raw_sum}, expected={expected_score}, "
            f"got={case.deterministic_score}"
        )

    for name, module in [
        ("metadata", case.metadata),
        ("font", case.font),
        ("compression", case.compression),
    ]:
        if module.score >= 5 and len(module.signals) == 0:
            warnings.append(f"⚠ FP RISK: {name} scores {module.score} with no signals")

    for name, module in [
        ("metadata", case.metadata),
        ("font", case.font),
        ("compression", case.compression),
    ]:
        if module.score == 0 and len(module.signals) >= 3:
            warnings.append(f"{name} scores 0 but has {len(module.signals)} signals")

    scores = [case.metadata.score, case.font.score, case.compression.score]
    non_zero = [s for s in scores if s > 0]
    if len(non_zero) == 1 and case.deterministic_score >= 5:
        warnings.append(
            f"⚠ FP RISK: Single module drives score {case.deterministic_score}"
        )

    if warnings:
        return "; ".join(warnings)
    return None


class CriticAgent:
    """
    Evidence quality auditor + false positive detector.

    LEARNING: Injects similar labeled cases into prompts
    so the Critic learns from real-world outcomes.
    """

    def __init__(self, client: Optional[LlamaClient] = None):
        self._client = client
        self._retriever = None

    @property
    def client(self) -> LlamaClient:
        if self._client is None:
            self._client = get_critic_client()
        return self._client

    @property
    def retriever(self):
        if self._retriever is None and settings.USE_DYNAMIC_FEWSHOTS:
            try:
                from app.memory.retriever import get_retriever
                self._retriever = get_retriever()
            except Exception as e:
                logger.warning(f"Retriever unavailable: {e}")
                self._retriever = None
        return self._retriever

    def _build_system_prompt(
        self,
        dynamic_examples: str = "",
    ) -> str:
        template = PromptLoader.get_template()

        # Combine static + dynamic examples
        static_examples = PromptLoader.get_fewshots()
        if static_examples:
            static_text = format_fewshot_examples(static_examples)
        else:
            static_text = ""

        if dynamic_examples and static_text:
            all_examples = f"{static_text}\n\n{dynamic_examples}"
        elif dynamic_examples:
            all_examples = dynamic_examples
        elif static_text:
            all_examples = static_text
        else:
            all_examples = "(No examples available — rely on instructions above.)"

        return template.replace("{few_shot_examples}", all_examples)

    def _build_user_prompt(
        self,
        case: EvidenceCaseFile,
        preflight_warning: Optional[str] = None,
    ) -> str:
        case_json = serialize_case_file(case)
        parts = [
            "Analyze the following case file.",
            "Assess consistency AND false positive risk.",
            "",
            "Case File:",
            case_json,
        ]
        if preflight_warning:
            parts.extend([
                "",
                "⚠ PREFLIGHT WARNINGS:",
                preflight_warning,
            ])
        return "\n".join(parts)

    async def audit(self, case: EvidenceCaseFile) -> CriticReport:
        logger.info(f"[{case.case_id}] Starting critic audit")

        # Preflight
        preflight_warning = preflight_check(case)
        if preflight_warning:
            logger.warning(f"[{case.case_id}] Preflight: {preflight_warning}")

        # Retrieve similar cases for dynamic few-shots
        dynamic_examples = ""
        if self.retriever:
            try:
                similar = self.retriever.retrieve_similar(case)
                if similar:
                    dynamic_examples = self.retriever.format_for_critic(similar)
                    logger.info(
                        f"[{case.case_id}] Injecting {len(similar)} dynamic examples"
                    )
            except Exception as e:
                logger.warning(f"[{case.case_id}] Retrieval failed: {e}")

        # Build prompts
        system_prompt = self._build_system_prompt(dynamic_examples)
        user_prompt = self._build_user_prompt(case, preflight_warning)

        # Call LLM
        try:
            report: CriticReport = await self.client.generate(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                schema=CriticReport,
            )
        except LLMFatalError:
            logger.error(f"[{case.case_id}] Critic FAILED")
            raise

        logger.info(
            f"[{case.case_id}] Critic complete | "
            f"FP_indicators={len(report.false_positive_indicators)} | "
            f"signal_strength={report.signal_strength.value} | "
            f"adjustment={report.recommended_adjustment.value} | "
            f"confidence={report.confidence:.2f}"
        )

        return report
