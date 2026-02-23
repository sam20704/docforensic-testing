"""
feedback/storage.py

Async SQLite storage for case outcomes and human feedback.
"""

import json
import logging
import aiosqlite
from pathlib import Path
from typing import Optional, List
from datetime import datetime

from app.config import settings
from app.feedback.schemas import StoredCase, FeedbackSubmission, LabeledExample


logger = logging.getLogger("forensic.feedback")


# ═══════════════════════════════════════════════════════
# SQL SCHEMA
# ═══════════════════════════════════════════════════════

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS case_feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    case_id TEXT NOT NULL UNIQUE,
    
    case_data TEXT NOT NULL,
    critic_report TEXT NOT NULL,
    verdict TEXT NOT NULL,
    
    judge_tampered BOOLEAN NOT NULL,
    judge_probability REAL NOT NULL,
    judge_severity TEXT NOT NULL,
    override_applied BOOLEAN NOT NULL,
    
    rule_score INTEGER NOT NULL,
    rule_priority TEXT NOT NULL,
    
    human_label TEXT DEFAULT NULL,
    human_notes TEXT DEFAULT NULL,
    reviewed_by TEXT DEFAULT NULL,
    reviewed_at TEXT DEFAULT NULL,
    
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    pipeline_duration_ms REAL DEFAULT NULL,
    
    embedded BOOLEAN DEFAULT 0
);
"""

CREATE_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS idx_human_label ON case_feedback(human_label);
CREATE INDEX IF NOT EXISTS idx_embedded ON case_feedback(embedded);
CREATE INDEX IF NOT EXISTS idx_override ON case_feedback(override_applied);
"""


# ═══════════════════════════════════════════════════════
# FEEDBACK STORE
# ═══════════════════════════════════════════════════════

class FeedbackStore:
    """
    Async SQLite store for case feedback.

    Stores:
        - Every case that goes through the pipeline
        - Human labels when reviewed
        - Learning status (embedded or not)
    """

    def __init__(self, db_path: Optional[str] = None):
        self._db_path = db_path or settings.FEEDBACK_DB_PATH
        self._initialized = False

    async def _ensure_db(self):
        """Create database and tables if they don't exist."""
        if self._initialized:
            return

        db_dir = Path(self._db_path).parent
        db_dir.mkdir(parents=True, exist_ok=True)

        async with aiosqlite.connect(self._db_path) as db:
            await db.executescript(CREATE_TABLE_SQL)
            await db.executescript(CREATE_INDEX_SQL)
            await db.commit()

        self._initialized = True
        logger.info(f"Feedback DB initialized: {self._db_path}")

    async def store_case(
        self,
        case_data: dict,
        critic_report: dict,
        verdict: dict,
        pipeline_duration_ms: float,
    ) -> None:
        """Store a completed pipeline result."""
        await self._ensure_db()

        case_id = case_data.get("case_id", "unknown")

        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                """
                INSERT OR REPLACE INTO case_feedback (
                    case_id, case_data, critic_report, verdict,
                    judge_tampered, judge_probability, judge_severity,
                    override_applied, rule_score, rule_priority,
                    pipeline_duration_ms, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    case_id,
                    json.dumps(case_data, default=str),
                    json.dumps(critic_report, default=str),
                    json.dumps(verdict, default=str),
                    verdict.get("tampered", False),
                    verdict.get("tampered_probability", 0.0),
                    verdict.get("severity", "Low"),
                    verdict.get("override_applied", False),
                    case_data.get("deterministic_score", 0),
                    case_data.get("priority", "Low"),
                    pipeline_duration_ms,
                    datetime.utcnow().isoformat(),
                ),
            )
            await db.commit()

        logger.info(f"[{case_id}] Case stored in feedback DB")

    async def submit_feedback(self, feedback: FeedbackSubmission) -> bool:
        """Record human feedback for a case."""
        await self._ensure_db()

        async with aiosqlite.connect(self._db_path) as db:
            cursor = await db.execute(
                "SELECT id FROM case_feedback WHERE case_id = ?",
                (feedback.case_id,),
            )
            row = await cursor.fetchone()

            if not row:
                logger.warning(f"[{feedback.case_id}] Case not found for feedback")
                return False

            await db.execute(
                """
                UPDATE case_feedback SET
                    human_label = ?,
                    human_notes = ?,
                    reviewed_by = ?,
                    reviewed_at = ?
                WHERE case_id = ?
                """,
                (
                    feedback.human_label.value,
                    feedback.human_notes,
                    feedback.reviewed_by,
                    datetime.utcnow().isoformat(),
                    feedback.case_id,
                ),
            )
            await db.commit()

        logger.info(
            f"[{feedback.case_id}] Feedback recorded: "
            f"label={feedback.human_label.value}"
        )
        return True

    async def get_labeled_cases(
        self,
        limit: int = 100,
    ) -> List[LabeledExample]:
        """Get all human-labeled cases for learning."""
        await self._ensure_db()

        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                """
                SELECT * FROM case_feedback
                WHERE human_label IS NOT NULL
                ORDER BY reviewed_at DESC
                LIMIT ?
                """,
                (limit,),
            )
            rows = await cursor.fetchall()

        examples = []
        for row in rows:
            examples.append(LabeledExample(
                case_id=row["case_id"],
                case_data=json.loads(row["case_data"]),
                judge_tampered=bool(row["judge_tampered"]),
                judge_probability=row["judge_probability"],
                judge_severity=row["judge_severity"],
                override_applied=bool(row["override_applied"]),
                rule_score=row["rule_score"],
                rule_priority=row["rule_priority"],
                human_label=row["human_label"],
                human_notes=row["human_notes"],
            ))

        logger.info(f"Retrieved {len(examples)} labeled cases")
        return examples

    async def get_unlabeled_cases(
        self,
        limit: int = 50,
    ) -> List[dict]:
        """Get cases pending human review."""
        await self._ensure_db()

        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                """
                SELECT case_id, case_data, verdict,
                       judge_tampered, judge_probability, judge_severity,
                       override_applied, rule_score, rule_priority,
                       created_at
                FROM case_feedback
                WHERE human_label IS NULL
                ORDER BY
                    override_applied DESC,
                    created_at DESC
                LIMIT ?
                """,
                (limit,),
            )
            rows = await cursor.fetchall()

        cases = []
        for row in rows:
            cases.append({
                "case_id": row["case_id"],
                "case_data": json.loads(row["case_data"]),
                "verdict": json.loads(row["verdict"]),
                "judge_tampered": bool(row["judge_tampered"]),
                "judge_probability": row["judge_probability"],
                "judge_severity": row["judge_severity"],
                "override_applied": bool(row["override_applied"]),
                "rule_score": row["rule_score"],
                "rule_priority": row["rule_priority"],
                "created_at": row["created_at"],
            })

        return cases

    async def mark_embedded(self, case_id: str) -> None:
        """Mark a case as embedded in vector store."""
        await self._ensure_db()

        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                "UPDATE case_feedback SET embedded = 1 WHERE case_id = ?",
                (case_id,),
            )
            await db.commit()

    async def get_stats(self) -> dict:
        """Get feedback database statistics."""
        await self._ensure_db()

        async with aiosqlite.connect(self._db_path) as db:
            total = await (await db.execute(
                "SELECT COUNT(*) FROM case_feedback"
            )).fetchone()
            labeled = await (await db.execute(
                "SELECT COUNT(*) FROM case_feedback WHERE human_label IS NOT NULL"
            )).fetchone()
            overrides = await (await db.execute(
                "SELECT COUNT(*) FROM case_feedback WHERE override_applied = 1"
            )).fetchone()
            embedded = await (await db.execute(
                "SELECT COUNT(*) FROM case_feedback WHERE embedded = 1"
            )).fetchone()

            # Accuracy (where labeled)
            correct = await (await db.execute(
                """
                SELECT COUNT(*) FROM case_feedback
                WHERE (human_label = 'tampered' AND judge_tampered = 1)
                   OR (human_label = 'not_tampered' AND judge_tampered = 0)
                """
            )).fetchone()

        total_count = total[0]
        labeled_count = labeled[0]

        return {
            "total_cases": total_count,
            "labeled_cases": labeled_count,
            "unlabeled_cases": total_count - labeled_count,
            "override_cases": overrides[0],
            "embedded_cases": embedded[0],
            "accuracy": (
                round(correct[0] / labeled_count, 3)
                if labeled_count > 0 else None
            ),
        }


# ── Singleton ──

_store: Optional[FeedbackStore] = None


def get_feedback_store() -> FeedbackStore:
    global _store
    if _store is None:
        _store = FeedbackStore()
        logger.info("FeedbackStore initialized")
    return _store
