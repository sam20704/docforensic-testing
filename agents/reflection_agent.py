"""
agents/reflection_agent.py

Learned Judge Agent — with case memory.

LEARNING UPGRADE: Sees similar past cases (with verified outcomes)
before making decisions. Learns from past mistakes.
"""

import json
import logging
from pathlib import Path
from typing import Optional

from app.schemas.evidence import EvidenceCaseFile
from app.schemas.critic import CriticReport
from app.schemas.reflection import ForensicVerdict
from app.services.llm_clients import ClaudeClient, get_reflection_client, LLMFatalError
from app.config import settings


logger = logging.getLogger("forensic.judge")

_PROMPT_DIR = Path(__file__).resolve().parent.parent / "prompts"
_PROMPT_FILE = _PROMPT_DIR / "reflection_prompt.txt"


class ReflectionPromptLoader:
    _template: Optional[str] = None

    @classmethod
    def get_template(cls) -> str:
        if cls._template is None:
            if not _PROMPT_FILE.exists():
                raise FileNotFoundError(f"Reflection prompt not found: {_PROMPT_FILE}")
            cls._template = _PROMPT_FILE.read_text(encoding="utf-8")
            logger.info(f"Loaded reflection prompt ({len(cls._template)} chars)")
        return cls._template

    @classmethod
    def clear_cache(cls):
        cls._template = None


def build_reflection_input(
    case: EvidenceCaseFile,
    critic_report: CriticReport,
) -> str:
    combined = {
        "case_file": case.model_dump(exclude_none=True),
        "critic_report": critic_report.model_dump(),
    }
    return json.dumps(combined, indent=2, default=str)


def should_flag_for_review(
    case: EvidenceCaseFile,
    critic: CriticReport,
) -> Optional[str]:
    reasons = []

    if critic.confidence < 0.5:
        reasons.append(f"Critic confidence low ({critic.confidence:.2f})")
    if critic.rerun_recommended:
        reasons.append("Rerun recommended")

    high_contra = [c for c in critic.contradictions if c.impact.value == "High"]
    if high_contra:
        reasons.append(f"{len(high_contra)} high-impact contradiction(s)")
    if case.deterministic_score in (4, 5):
        reasons.append(f"Borderline score ({case.deterministic_score})")

    modules_with_signals = sum(
        1 for m in [case.metadata, case.font, case.compression]
        if len(m.signals) > 0
    )
    if modules_with_signals <= 1 and case.deterministic_score >= 4:
        reasons.append("Sparse evidence")
    if not critic.rule_consistency:
        reasons.append("Rule inconsistency")
    if critic.evidence_sufficiency.value in ("insufficient", "borderline"):
        reasons.append(f"Evidence: {critic.evidence_sufficiency.value}")

    return "; ".join(reasons) if reasons else None


def audit_verdict(
    verdict: ForensicVerdict,
    case: EvidenceCaseFile,
    review_reason: Optional[str],
) -> ForensicVerdict:
    corrections = []

    if verdict.case_id != case.case_id:
        corrections.append(f"Fixed case_id: {verdict.case_id} → {case.case_id}")
        verdict.case_id = case.case_id

    if verdict.rule_based_score != case.deterministic_score:
        corrections.append(f"Fixed rule_based_score → {case.deterministic_score}")
        verdict.rule_based_score = case.deterministic_score

    if verdict.rule_based_priority != case.priority.value:
        corrections.append(f"Fixed rule_based_priority → {case.priority.value}")
        verdict.rule_based_priority = case.priority.value

    # Auto-detect override
    if case.priority.value != verdict.severity.value and not verdict.override_applied:
        verdict.override_applied = True
        if not verdict.override_reason:
            verdict.override_reason = (
                f"Changed from {case.priority.value} to {verdict.severity.value}"
            )
        corrections.append("Auto-detected override")

    # Force review from pre-check
    if review_reason and not verdict.flagged_for_human_review:
        verdict.flagged_for_human_review = True
        verdict.review_reason = review_reason
        corrections.append("Forced review flag")

    # Flag low-confidence overrides
    if verdict.override_applied and not verdict.flagged_for_human_review:
        if verdict.confidence < 0.8:
            verdict.flagged_for_human_review = True
            verdict.review_reason = (
                f"Override with confidence {verdict.confidence:.2f}"
            )
            corrections.append("Flagged low-confidence override")

    if corrections:
        logger.warning(f"[{case.case_id}] Audit: {'; '.join(corrections)}")

    if verdict.override_applied:
        logger.info(
            f"[{case.case_id}] ⚡ OVERRIDE: "
            f"{case.priority.value}→{verdict.severity.value} "
            f"(prob={verdict.tampered_probability:.2f})"
        )

    return verdict


class ReflectionAgent:
    """
    Learned Judge — with case memory injection.

    Sees similar past cases before deciding, learning
    from verified human outcomes.
    """

    def __init__(self, client: Optional[ClaudeClient] = None):
        self._client = client
        self._retriever = None

    @property
    def client(self) -> ClaudeClient:
        if self._client is None:
            self._client = get_reflection_client()
        return self._client

    @property
    def retriever(self):
        if self._retriever is None and settings.USE_DYNAMIC_FEWSHOTS:
            try:
                from app.memory.retriever import get_retriever
                self._retriever = get_retriever()
            except Exception as e:
                logger.warning(f"Retriever unavailable: {e}")
        return self._retriever

    def _build_system_prompt(self) -> str:
        return ReflectionPromptLoader.get_template()

    def _build_user_prompt(
        self,
        case: EvidenceCaseFile,
        critic_report: CriticReport,
        review_hint: Optional[str] = None,
        similar_cases_text: str = "",
    ) -> str:
        combined_json = build_reflection_input(case, critic_report)

        parts = [
            "Review the following case file and critic report.",
            "Make your independent assessment.",
            "You CAN override rule-based scores if evidence warrants it.",
            "",
            combined_json,
        ]

        # Inject similar cases (THE LEARNING)
        if similar_cases_text:
            parts.extend(["", similar_cases_text])

        if review_hint:
            parts.extend([
                "",
                "⚠ PRE-CHECK FLAGS:",
                review_hint,
            ])

        parts.extend([
            "",
            "FIELD REMINDERS:",
            f"- case_id MUST be: {case.case_id}",
            f"- rule_based_score MUST be: {case.deterministic_score}",
            f"- rule_based_priority MUST be: {case.priority.value}",
            "",
            "severity and tampered are YOUR decision.",
        ])

        return "\n".join(parts)

    async def judge(
        self,
        case: EvidenceCaseFile,
        critic_report: CriticReport,
    ) -> ForensicVerdict:
        logger.info(f"[{case.case_id}] Starting learned judgment")

        review_reason = should_flag_for_review(case, critic_report)
        if review_reason:
            logger.info(f"[{case.case_id}] Pre-check flags: {review_reason}")

        # Retrieve similar cases
        similar_cases_text = ""
        if self.retriever:
            try:
                similar = self.retriever.retrieve_similar(case)
                if similar:
                    similar_cases_text = self.retriever.format_for_judge(similar)
                    logger.info(
                        f"[{case.case_id}] Injecting {len(similar)} similar cases for Judge"
                    )
            except Exception as e:
                logger.warning(f"[{case.case_id}] Retrieval failed: {e}")

        system_prompt = self._build_system_prompt()
        user_prompt = self._build_user_prompt(
            case, critic_report, review_reason, similar_cases_text
        )

        try:
            verdict: ForensicVerdict = await self.client.generate(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                schema=ForensicVerdict,
            )
        except LLMFatalError:
            logger.error(f"[{case.case_id}] Judge FAILED")
            raise

        verdict = audit_verdict(verdict, case, review_reason)

        logger.info(
            f"[{case.case_id}] Verdict | "
            f"tampered={verdict.tampered} | "
            f"prob={verdict.tampered_probability:.2f} | "
            f"severity={verdict.severity.value} | "
            f"override={verdict.override_applied} | "
            f"flagged={verdict.flagged_for_human_review}"
        )

        return verdict
