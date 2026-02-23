"""
services/validator.py

Forensic Validation Orchestrator.

LEARNING UPGRADE: After each verdict, automatically stores
the case in feedback DB and embeds it in vector store.
"""

import time
import logging
from typing import Optional
from dataclasses import dataclass

from app.schemas.evidence import EvidenceCaseFile
from app.schemas.critic import CriticReport
from app.schemas.reflection import ForensicVerdict
from app.agents.critic_agent import CriticAgent
from app.agents.reflection_agent import ReflectionAgent
from app.services.llm_clients import LLMFatalError


logger = logging.getLogger("forensic.validator")


# ═══════════════════════════════════════════════════════
# PIPELINE RESULT
# ═══════════════════════════════════════════════════════

@dataclass
class PipelineResult:
    verdict: ForensicVerdict
    critic_report: CriticReport
    case_id: str
    duration_ms: float
    success: bool
    similar_cases_used: int = 0
    error: Optional[str] = None
    failed_stage: Optional[str] = None

    def to_dict(self) -> dict:
        result = {
            "case_id": self.case_id,
            "success": self.success,
            "duration_ms": round(self.duration_ms, 2),
            "similar_cases_used": self.similar_cases_used,
            "verdict": self.verdict.model_dump(mode="json"),
            "critic_report": self.critic_report.model_dump(mode="json"),
        }
        if self.error:
            result["error"] = self.error
            result["failed_stage"] = self.failed_stage
        return result


# ═══════════════════════════════════════════════════════
# PIPELINE ERROR
# ═══════════════════════════════════════════════════════

class PipelineError(Exception):
    def __init__(self, case_id: str, stage: str, cause: Exception):
        self.case_id = case_id
        self.stage = stage
        self.cause = cause
        super().__init__(f"[{case_id}] Pipeline failed at {stage}: {cause}")


# ═══════════════════════════════════════════════════════
# FORENSIC VALIDATOR (Orchestrator)
# ═══════════════════════════════════════════════════════

class ForensicValidator:
    """
    Pipeline orchestrator with learning integration.

    Flow:
        1. Critic audits evidence + assesses FP risk
        2. Judge makes final decision (CAN override rules)
        3. Store case in feedback DB (for human review)
        4. Embed case in vector store (for future retrieval)

    Steps 3-4 are non-blocking — pipeline doesn't fail
    if storage fails. Verdict is returned regardless.
    """

    def __init__(
        self,
        critic: Optional[CriticAgent] = None,
        reflection: Optional[ReflectionAgent] = None,
    ):
        self._critic = critic or CriticAgent()
        self._reflection = reflection or ReflectionAgent()

    async def validate(self, case: EvidenceCaseFile) -> PipelineResult:
        start_time = time.monotonic()

        logger.info(
            f"[{case.case_id}] ═══ Pipeline started ═══ | "
            f"score={case.deterministic_score} | "
            f"priority={case.priority.value}"
        )

        # ── Stage 1: Critic ──
        critic_report = await self._run_critic(case)

        # ── Stage 2: Judge ──
        verdict = await self._run_judge(case, critic_report)

        # ── Compute duration ──
        duration_ms = (time.monotonic() - start_time) * 1000

        # ── Stage 3: Store + Embed (non-blocking) ──
        await self._store_and_embed(case, critic_report, verdict, duration_ms)

        # ── Package result ──
        result = PipelineResult(
            verdict=verdict,
            critic_report=critic_report,
            case_id=case.case_id,
            duration_ms=duration_ms,
            success=True,
        )

        logger.info(
            f"[{case.case_id}] ═══ Pipeline complete ═══ | "
            f"tampered={verdict.tampered} | "
            f"probability={verdict.tampered_probability:.2f} | "
            f"severity={verdict.severity.value} | "
            f"override={verdict.override_applied} | "
            f"flagged={verdict.flagged_for_human_review} | "
            f"duration={duration_ms:.0f}ms"
        )

        return result

    async def _run_critic(self, case: EvidenceCaseFile) -> CriticReport:
        """Execute critic stage with error handling."""
        logger.info(f"[{case.case_id}] Stage 1/2: Critic audit + FP assessment")

        try:
            return await self._critic.audit(case)

        except LLMFatalError as e:
            logger.error(f"[{case.case_id}] Critic FAILED: {e}")
            raise PipelineError(case_id=case.case_id, stage="critic", cause=e)

        except FileNotFoundError as e:
            logger.error(f"[{case.case_id}] Critic prompt missing: {e}")
            raise PipelineError(case_id=case.case_id, stage="critic", cause=e)

        except Exception as e:
            logger.error(
                f"[{case.case_id}] Critic unexpected error: {e}",
                exc_info=True,
            )
            raise PipelineError(case_id=case.case_id, stage="critic", cause=e)

    async def _run_judge(
        self,
        case: EvidenceCaseFile,
        critic_report: CriticReport,
    ) -> ForensicVerdict:
        """Execute judge stage with error handling."""
        logger.info(f"[{case.case_id}] Stage 2/2: Learned judgment")

        try:
            return await self._reflection.judge(case, critic_report)

        except LLMFatalError as e:
            logger.error(f"[{case.case_id}] Judge FAILED: {e}")
            raise PipelineError(case_id=case.case_id, stage="judge", cause=e)

        except FileNotFoundError as e:
            logger.error(f"[{case.case_id}] Judge prompt missing: {e}")
            raise PipelineError(case_id=case.case_id, stage="judge", cause=e)

        except Exception as e:
            logger.error(
                f"[{case.case_id}] Judge unexpected error: {e}",
                exc_info=True,
            )
            raise PipelineError(case_id=case.case_id, stage="judge", cause=e)

    async def _store_and_embed(
        self,
        case: EvidenceCaseFile,
        critic_report: CriticReport,
        verdict: ForensicVerdict,
        duration_ms: float,
    ) -> None:
        """
        Store case in feedback DB + embed in vector store.

        This is NON-BLOCKING — if storage fails, the verdict
        is still returned. Learning is best-effort.
        """
        try:
            await self._store_feedback(case, critic_report, verdict, duration_ms)
        except Exception as e:
            logger.warning(
                f"[{case.case_id}] Feedback storage failed (non-fatal): {e}"
            )

        try:
            self._embed_case(case, verdict)
        except Exception as e:
            logger.warning(
                f"[{case.case_id}] Vector embedding failed (non-fatal): {e}"
            )

    async def _store_feedback(
        self,
        case: EvidenceCaseFile,
        critic_report: CriticReport,
        verdict: ForensicVerdict,
        duration_ms: float,
    ) -> None:
        """Store case in feedback DB for human review."""
        try:
            from app.feedback.storage import get_feedback_store

            store = get_feedback_store()
            await store.store_case(
                case_data=case.model_dump(exclude_none=True),
                critic_report=critic_report.model_dump(mode="json"),
                verdict=verdict.model_dump(mode="json"),
                pipeline_duration_ms=duration_ms,
            )

            logger.debug(f"[{case.case_id}] Stored in feedback DB")

        except ImportError:
            logger.debug("Feedback storage not available (aiosqlite not installed)")

    def _embed_case(
        self,
        case: EvidenceCaseFile,
        verdict: ForensicVerdict,
    ) -> None:
        """Embed case in vector store for similarity retrieval."""
        try:
            from app.memory.embedder import case_with_outcome_to_text
            from app.memory.vector_store import get_vector_store

            store = get_vector_store()

            # Build text representation
            text = case_with_outcome_to_text(
                case_data=case.model_dump(exclude_none=True),
                judge_tampered=verdict.tampered,
                judge_severity=verdict.severity.value,
                human_label=None,  # Not labeled yet
            )

            # Build metadata for filtering
            metadata = {
                "case_id": case.case_id,
                "rule_score": case.deterministic_score,
                "rule_priority": case.priority.value,
                "judge_tampered": str(verdict.tampered),
                "judge_probability": verdict.tampered_probability,
                "judge_severity": verdict.severity.value,
                "override_applied": str(verdict.override_applied),
                "metadata_score": case.metadata.score,
                "metadata_signals": ",".join(case.metadata.signals),
                "font_score": case.font.score,
                "font_signals": ",".join(case.font.signals),
                "compression_score": case.compression.score,
                "compression_signals": ",".join(case.compression.signals),
                "human_label": "",  # Empty until reviewed
                "human_notes": "",
            }

            store.add_case(
                case_id=case.case_id,
                text=text,
                metadata=metadata,
            )

            logger.debug(f"[{case.case_id}] Embedded in vector store")

        except ImportError:
            logger.debug("Vector store not available (chromadb not installed)")


# ═══════════════════════════════════════════════════════
# EMBEDDING UPDATER (Called after human feedback)
# ═══════════════════════════════════════════════════════

async def update_embedding_after_feedback(
    case_id: str,
    human_label: str,
    human_notes: Optional[str] = None,
) -> None:
    """
    Update vector store embedding after human feedback is received.

    This is what makes the system LEARN:
    - Case gets human label
    - Embedding is updated with the label
    - Future similar cases will see this labeled case
    - Agents make better decisions

    Called from the /feedback endpoint.
    """
    try:
        from app.feedback.storage import get_feedback_store
        from app.memory.embedder import case_with_outcome_to_text
        from app.memory.vector_store import get_vector_store

        # Get the stored case data
        store = get_feedback_store()
        feedback_store = store

        # Get case from DB
        import aiosqlite
        import json
        from app.config import settings

        async with aiosqlite.connect(settings.FEEDBACK_DB_PATH) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT * FROM case_feedback WHERE case_id = ?",
                (case_id,),
            )
            row = await cursor.fetchone()

        if not row:
            logger.warning(f"[{case_id}] Case not found for embedding update")
            return

        case_data = json.loads(row["case_data"])

        # Rebuild embedding WITH human label
        text = case_with_outcome_to_text(
            case_data=case_data,
            judge_tampered=bool(row["judge_tampered"]),
            judge_severity=row["judge_severity"],
            human_label=human_label,
        )

        # Update metadata with human label
        metadata = {
            "case_id": case_id,
            "rule_score": row["rule_score"],
            "rule_priority": row["rule_priority"],
            "judge_tampered": str(bool(row["judge_tampered"])),
            "judge_probability": row["judge_probability"],
            "judge_severity": row["judge_severity"],
            "override_applied": str(bool(row["override_applied"])),
            "metadata_score": case_data.get("metadata", {}).get("score", 0),
            "metadata_signals": ",".join(
                case_data.get("metadata", {}).get("signals", [])
            ),
            "font_score": case_data.get("font", {}).get("score", 0),
            "font_signals": ",".join(
                case_data.get("font", {}).get("signals", [])
            ),
            "compression_score": case_data.get("compression", {}).get("score", 0),
            "compression_signals": ",".join(
                case_data.get("compression", {}).get("signals", [])
            ),
            "human_label": human_label,
            "human_notes": human_notes or "",
        }

        # Upsert in vector store
        vector_store = get_vector_store()
        vector_store.add_case(
            case_id=case_id,
            text=text,
            metadata=metadata,
        )

        # Mark as embedded in feedback DB
        await feedback_store.mark_embedded(case_id)

        logger.info(
            f"[{case_id}] Embedding updated with human label: {human_label}"
        )

    except ImportError as e:
        logger.warning(f"[{case_id}] Embedding update skipped (missing dependency): {e}")

    except Exception as e:
        logger.error(
            f"[{case_id}] Embedding update failed: {e}",
            exc_info=True,
        )


# ═══════════════════════════════════════════════════════
# CONVENIENCE FUNCTIONS
# ═══════════════════════════════════════════════════════

_validator: Optional[ForensicValidator] = None


def get_validator() -> ForensicValidator:
    """Get or create singleton ForensicValidator."""
    global _validator
    if _validator is None:
        _validator = ForensicValidator()
        logger.info("ForensicValidator initialized")
    return _validator


async def validate_case(case: EvidenceCaseFile) -> PipelineResult:
    """One-liner convenience function."""
    validator = get_validator()
    return await validator.validate(case)
