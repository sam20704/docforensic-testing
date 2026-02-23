"""
agents/reflection_agent.py

Learned Judge Agent — Final decision maker.

Uses Claude to:
    - Evaluate ALL evidence (rules are features, not authority)
    - Produce independent tampered probability
    - Override rule-based scores when warranted
    - Explain reasoning for audit trail

KEY CHANGE: The Judge CAN and SHOULD override false positives.
The old validate_verdict() that forced scores back is REMOVED.
"""

import json
import logging
from pathlib import Path
from typing import Optional

from app.schemas.evidence import EvidenceCaseFile
from app.schemas.critic import CriticReport
from app.schemas.reflection import ForensicVerdict
from app.services.llm_clients import ClaudeClient, get_reflection_client, LLMFatalError


logger = logging.getLogger("forensic.judge")

_PROMPT_DIR = Path(__file__).resolve().parent.parent / "prompts"
_PROMPT_FILE = _PROMPT_DIR / "reflection_prompt.txt"


# ═══════════════════════════════════════════════════════
# PROMPT LOADER
# ═══════════════════════════════════════════════════════

class ReflectionPromptLoader:
    _template: Optional[str] = None

    @classmethod
    def get_template(cls) -> str:
        if cls._template is None:
            if not _PROMPT_FILE.exists():
                raise FileNotFoundError(
                    f"Reflection prompt not found: {_PROMPT_FILE}"
                )
            cls._template = _PROMPT_FILE.read_text(encoding="utf-8")
            logger.info(f"Loaded reflection prompt ({len(cls._template)} chars)")
        return cls._template

    @classmethod
    def clear_cache(cls):
        cls._template = None


# ═══════════════════════════════════════════════════════
# INPUT SERIALIZER
# ═══════════════════════════════════════════════════════

def build_reflection_input(
    case: EvidenceCaseFile,
    critic_report: CriticReport,
) -> str:
    case_data = case.model_dump(exclude_none=True)
    critic_data = critic_report.model_dump()
    combined = {
        "case_file": case_data,
        "critic_report": critic_data,
    }
    return json.dumps(combined, indent=2, default=str)


# ═══════════════════════════════════════════════════════
# PRE-VERDICT REVIEW FLAGS
# ═══════════════════════════════════════════════════════

def should_flag_for_review(
    case: EvidenceCaseFile,
    critic: CriticReport,
) -> Optional[str]:
    """
    Deterministic pre-check for human review.

    Updated: now also flags when override is likely,
    so humans can verify the LLM's override decisions.
    """
    reasons = []

    if critic.confidence < 0.5:
        reasons.append(f"Critic confidence is low ({critic.confidence:.2f})")

    if critic.rerun_recommended:
        reasons.append("Critic recommended deterministic rerun")

    high_contradictions = [
        c for c in critic.contradictions if c.impact.value == "High"
    ]
    if high_contradictions:
        reasons.append(f"{len(high_contradictions)} high-impact contradiction(s)")

    if case.deterministic_score in (4, 5):
        reasons.append(f"Borderline score ({case.deterministic_score})")

    modules_with_signals = sum(
        1 for m in [case.metadata, case.font, case.compression]
        if len(m.signals) > 0
    )
    if modules_with_signals <= 1 and case.deterministic_score >= 4:
        reasons.append(
            f"Sparse evidence: {modules_with_signals} module(s) with signals "
            f"for score {case.deterministic_score}"
        )

    if not critic.rule_consistency:
        reasons.append("Critic detected rule inconsistency")

    if critic.evidence_sufficiency.value in ("insufficient", "borderline"):
        reasons.append(
            f"Evidence sufficiency: {critic.evidence_sufficiency.value}"
        )

    if reasons:
        return "; ".join(reasons)
    return None


# ═══════════════════════════════════════════════════════
# POST-VERDICT AUDIT (NOT enforcement)
# ═══════════════════════════════════════════════════════

def audit_verdict(
    verdict: ForensicVerdict,
    case: EvidenceCaseFile,
    review_reason: Optional[str],
) -> ForensicVerdict:
    """
    Post-verdict audit: log overrides, fix ONLY passthrough fields.

    KEY CHANGE from old validate_verdict():
    ❌ We do NOT force severity back to rules
    ❌ We do NOT force deterministic_score
    ✅ We DO fix case_id (identity, not judgment)
    ✅ We DO fix rule_based_score/priority (reference fields)
    ✅ We DO enforce review flag when pre-check says so
    ✅ We DO log when override happens (audit trail)
    """
    corrections = []

    # Fix case_id (identity passthrough, not a judgment)
    if verdict.case_id != case.case_id:
        corrections.append(f"Fixed case_id: {verdict.case_id} → {case.case_id}")
        verdict.case_id = case.case_id

    # Fix rule_based reference fields (these are READ-ONLY references)
    if verdict.rule_based_score != case.deterministic_score:
        corrections.append(
            f"Fixed rule_based_score: {verdict.rule_based_score} → {case.deterministic_score}"
        )
        verdict.rule_based_score = case.deterministic_score

    if verdict.rule_based_priority != case.priority.value:
        corrections.append(
            f"Fixed rule_based_priority: {verdict.rule_based_priority} → {case.priority.value}"
        )
        verdict.rule_based_priority = case.priority.value

    # Detect override (in case LLM didn't set the flag correctly)
    rules_severity = case.priority.value
    judge_severity = verdict.severity.value
    if rules_severity != judge_severity and not verdict.override_applied:
        verdict.override_applied = True
        if not verdict.override_reason:
            verdict.override_reason = (
                f"Judge changed severity from {rules_severity} to {judge_severity}"
            )
        corrections.append("Auto-detected override (LLM forgot to set flag)")

    # Force review flag from deterministic pre-check
    if review_reason and not verdict.flagged_for_human_review:
        corrections.append("Forced review flag from pre-check")
        verdict.flagged_for_human_review = True
        verdict.review_reason = review_reason

    # Flag overrides for review (trust but verify, especially early on)
    if verdict.override_applied and not verdict.flagged_for_human_review:
        if verdict.confidence < 0.8:
            verdict.flagged_for_human_review = True
            verdict.review_reason = (
                f"Override applied with confidence {verdict.confidence:.2f} — "
                f"flagging for human verification"
            )
            corrections.append("Flagged override for human review (confidence < 0.8)")

    if corrections:
        logger.warning(
            f"[{case.case_id}] Post-verdict audit: {'; '.join(corrections)}"
        )

    # Log override details (audit trail)
    if verdict.override_applied:
        logger.info(
            f"[{case.case_id}] ⚡ OVERRIDE: "
            f"Rules said {case.priority.value} (score {case.deterministic_score}) → "
            f"Judge says {verdict.severity.value} "
            f"(probability {verdict.tampered_probability:.2f}) | "
            f"Reason: {verdict.override_reason}"
        )

    return verdict


# ═══════════════════════════════════════════════════════
# REFLECTION AGENT (Learned Judge)
# ═══════════════════════════════════════════════════════

class ReflectionAgent:
    """
    Learned Judge — makes the final tampering decision.

    CAN override rule-based scores.
    CAN assign different severity.
    CAN say "this is a false positive."

    The old handcuffs (validate_verdict forcing scores back)
    are REMOVED. The Judge has full authority.
    """

    def __init__(self, client: Optional[ClaudeClient] = None):
        self._client = client

    @property
    def client(self) -> ClaudeClient:
        if self._client is None:
            self._client = get_reflection_client()
        return self._client

    def _build_system_prompt(self) -> str:
        return ReflectionPromptLoader.get_template()

    def _build_user_prompt(
        self,
        case: EvidenceCaseFile,
        critic_report: CriticReport,
        review_hint: Optional[str] = None,
    ) -> str:
        combined_json = build_reflection_input(case, critic_report)

        parts = [
            "Review the following case file and critic audit report.",
            "Make your independent assessment.",
            "You CAN override the rule-based scores if evidence warrants it.",
            "",
            combined_json,
        ]

        if review_hint:
            parts.extend([
                "",
                "⚠ PRE-CHECK FLAGS:",
                review_hint,
                "",
                "Consider flagging for human review.",
            ])

        # Passthrough reminders (identity fields only, NOT judgment constraints)
        parts.extend([
            "",
            "FIELD REMINDERS:",
            f"- case_id MUST be: {case.case_id}",
            f"- rule_based_score MUST be: {case.deterministic_score}",
            f"- rule_based_priority MUST be: {case.priority.value}",
            "",
            "NOTE: severity and tampered are YOUR decision.",
            "You are NOT required to match rule_based_priority.",
        ])

        return "\n".join(parts)

    async def judge(
        self,
        case: EvidenceCaseFile,
        critic_report: CriticReport,
    ) -> ForensicVerdict:
        logger.info(f"[{case.case_id}] Starting learned judgment")

        # Pre-verdict review check
        review_reason = should_flag_for_review(case, critic_report)
        if review_reason:
            logger.info(f"[{case.case_id}] Pre-check flags: {review_reason}")

        system_prompt = self._build_system_prompt()
        user_prompt = self._build_user_prompt(case, critic_report, review_reason)

        logger.debug(
            f"[{case.case_id}] System: {len(system_prompt)} chars | "
            f"User: {len(user_prompt)} chars"
        )

        try:
            verdict: ForensicVerdict = await self.client.generate(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                schema=ForensicVerdict,
            )
        except LLMFatalError:
            logger.error(f"[{case.case_id}] Judge FAILED — LLM fatal error")
            raise

        # Post-verdict audit (NOT enforcement — Judge has authority)
        verdict = audit_verdict(verdict, case, review_reason)

        logger.info(
            f"[{case.case_id}] Verdict | "
            f"tampered={verdict.tampered} | "
            f"probability={verdict.tampered_probability:.2f} | "
            f"severity={verdict.severity.value} | "
            f"override={verdict.override_applied} | "
            f"confidence={verdict.confidence:.2f} | "
            f"flagged={verdict.flagged_for_human_review}"
        )

        return verdict
