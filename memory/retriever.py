"""
memory/retriever.py

Retrieves similar labeled cases and formats them
for injection into agent prompts.

This is where LEARNING happens:
    1. New case comes in
    2. Find similar past cases (that humans have labeled)
    3. Format them as examples
    4. Inject into Critic + Judge prompts
    5. Agents see real outcomes → make better decisions
"""

import logging
from typing import List, Optional

from app.schemas.evidence import EvidenceCaseFile
from app.memory.embedder import case_to_text
from app.memory.vector_store import get_vector_store
from app.feedback.schemas import LabeledExample
from app.config import settings


logger = logging.getLogger("forensic.retriever")


class CaseRetriever:
    """
    Retrieves similar labeled cases for dynamic few-shot injection.

    Only returns cases that:
        - Have a human label (confirmed outcome)
        - Are above the similarity threshold
        - Are sorted by relevance
    """

    def __init__(self):
        self._store = get_vector_store()

    def retrieve_similar(
        self,
        case: EvidenceCaseFile,
        k: Optional[int] = None,
    ) -> List[LabeledExample]:
        """
        Find similar labeled cases for the given input.

        Args:
            case: The new case to find matches for
            k: Max results (defaults to config)

        Returns:
            List of LabeledExample, sorted by similarity (most similar first)
        """
        k = k or settings.SIMILAR_CASES_K

        if self._store.count() == 0:
            logger.debug("Vector store is empty — no similar cases")
            return []

        # Embed the query case
        query_text = case_to_text(case)

        # Search for similar LABELED cases only
        results = self._store.search_similar(
            query_text=query_text,
            k=k,
            min_score=settings.SIMILAR_CASES_MIN_SCORE,
            where={"human_label": {"$ne": ""}},
        )

        if not results:
            logger.debug(f"[{case.case_id}] No similar labeled cases found")
            return []

        # Convert to LabeledExample
        examples = []
        for result in results:
            meta = result["metadata"]
            try:
                example = LabeledExample(
                    case_id=result["case_id"],
                    case_data={
                        "metadata": {
                            "score": int(meta.get("metadata_score", 0)),
                            "signals": meta.get("metadata_signals", "").split(",") if meta.get("metadata_signals") else [],
                        },
                        "font": {
                            "score": int(meta.get("font_score", 0)),
                            "signals": meta.get("font_signals", "").split(",") if meta.get("font_signals") else [],
                        },
                        "compression": {
                            "score": int(meta.get("compression_score", 0)),
                            "signals": meta.get("compression_signals", "").split(",") if meta.get("compression_signals") else [],
                        },
                        "deterministic_score": int(meta.get("rule_score", 0)),
                        "priority": meta.get("rule_priority", "Low"),
                    },
                    judge_tampered=meta.get("judge_tampered", "") == "True",
                    judge_probability=float(meta.get("judge_probability", 0)),
                    judge_severity=str(meta.get("judge_severity", "Low")),
                    override_applied=meta.get("override_applied", "") == "True",
                    rule_score=int(meta.get("rule_score", 0)),
                    rule_priority=str(meta.get("rule_priority", "Low")),
                    human_label=str(meta.get("human_label", "")),
                    human_notes=meta.get("human_notes", None) or None,
                    similarity_score=result["similarity_score"],
                )
                examples.append(example)
            except Exception as e:
                logger.warning(
                    f"Failed to parse similar case {result['case_id']}: {e}"
                )
                continue

        logger.info(
            f"[{case.case_id}] Retrieved {len(examples)} similar labeled cases "
            f"(similarities: {[e.similarity_score for e in examples]})"
        )

        return examples

    def format_for_critic(self, examples: List[LabeledExample]) -> str:
        """Format similar cases for Critic prompt injection."""
        if not examples:
            return ""

        blocks = [
            "═══ SIMILAR PAST CASES (with known outcomes) ═══",
            "Learn from these. Cases with similar patterns had these results:",
            "",
        ]

        for ex in examples:
            blocks.append(ex.to_prompt_text())

        blocks.append(
            "Use these past cases to calibrate your assessment. "
            "If similar cases were false positives, this one might be too."
        )

        return "\n".join(blocks)

    def format_for_judge(self, examples: List[LabeledExample]) -> str:
        """Format similar cases for Judge prompt injection."""
        if not examples:
            return ""

        # Separate correct and incorrect past decisions
        correct = [e for e in examples if e.was_correct()]
        incorrect = [e for e in examples if not e.was_correct()]

        blocks = [
            "═══ SIMILAR PAST CASES (verified by humans) ═══",
            "",
        ]

        if correct:
            blocks.append("Cases where the Judge was CORRECT:")
            for ex in correct:
                blocks.append(ex.to_prompt_text())

        if incorrect:
            blocks.append("Cases where the Judge was WRONG (learn from these):")
            for ex in incorrect:
                blocks.append(ex.to_prompt_text())
                if ex.human_notes:
                    blocks.append(f"  Human reviewer note: {ex.human_notes}")

        # FP-specific guidance from examples
        fp_examples = [
            e for e in examples
            if e.human_label == "not_tampered" and e.rule_score >= 5
        ]
        if fp_examples:
            blocks.append(
                f"\n⚠ NOTE: {len(fp_examples)} similar case(s) with high rule scores "
                f"were confirmed NOT tampered by humans. Be cautious of false positives."
            )

        return "\n".join(blocks)


# ── Singleton ──

_retriever: Optional[CaseRetriever] = None


def get_retriever() -> CaseRetriever:
    global _retriever
    if _retriever is None:
        _retriever = CaseRetriever()
    return _retriever
