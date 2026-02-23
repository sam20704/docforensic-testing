"""
Offline deterministic tests for the adaptive forensic pipeline.
No LLM calls. No network. No optional dependencies.
"""

import pytest
import json
from unittest.mock import AsyncMock, MagicMock, patch

from pydantic import ValidationError

from app.schemas.evidence import EvidenceCaseFile, ModuleResult, Priority
from app.schemas.critic import (
    CriticReport,
    Contradiction,
    ContradictionType,
    ContradictionImpact,
    ConfidenceLevel as CriticConfidence,
    SignalStrength,
    EvidenceSufficiency,
    AdjustmentDirection,
)
from app.schemas.reflection import (
    ForensicVerdict,
    Severity,
    EvidenceItem,
    EvidenceSource,
    EvidenceWeight,
    ConfidenceLevel as JudgeConfidence,
)
from app.agents.critic_agent import preflight_check, serialize_case_file, PromptLoader
from app.agents.reflection_agent import should_flag_for_review, audit_verdict
from app.services.validator import (
    ForensicValidator,
    PipelineError,
    PipelineResult,
    reset_validator,
)
from app.services.llm_clients import LLMFatalError


# ═══════════════════════════════════════════════════════
# FIXTURES
# ═══════════════════════════════════════════════════════

@pytest.fixture(autouse=True)
def _reset_singleton():
    """Ensure clean validator state between tests."""
    reset_validator()
    yield
    reset_validator()


@pytest.fixture
def clean_case():
    return EvidenceCaseFile(
        case_id="CLEAN-001",
        metadata=ModuleResult(score=1, signals=["single_tool"]),
        font=ModuleResult(score=0, signals=[]),
        compression=ModuleResult(score=1, signals=["minor_artifact"]),
        deterministic_score=2,
        priority=Priority.Low,
    )


@pytest.fixture
def tampered_case():
    return EvidenceCaseFile(
        case_id="TAMPER-001",
        metadata=ModuleResult(score=6, signals=["tool_mismatch", "date_anomaly"]),
        font=ModuleResult(score=5, signals=["mixed_fonts", "subset_mismatch"]),
        compression=ModuleResult(score=4, signals=["double_compression"]),
        deterministic_score=10,
        priority=Priority.High,
    )


@pytest.fixture
def scanner_fp_case():
    """Classic false positive: scanner workflow triggers metadata."""
    return EvidenceCaseFile(
        case_id="SCANNER-FP-001",
        metadata=ModuleResult(score=6, signals=["multiple_tools_detected"]),
        font=ModuleResult(score=0, signals=[]),
        compression=ModuleResult(score=1, signals=["minor_artifact"]),
        deterministic_score=7,
        priority=Priority.Medium,
    )


def _build_critic_report(
    confidence: float = 0.9,
    signal_strength: SignalStrength = SignalStrength.MODERATE,
    sufficiency: EvidenceSufficiency = EvidenceSufficiency.SUFFICIENT,
    adjustment: AdjustmentDirection = AdjustmentDirection.MAINTAIN,
    fp_indicators: list = None,
    rerun: bool = False,
    contradictions: list = None,
    rule_consistency: bool = True,
) -> CriticReport:
    """Helper: build a valid CriticReport with correct confidence_level."""
    if confidence >= 0.9:
        level = CriticConfidence.VERY_HIGH
    elif confidence >= 0.7:
        level = CriticConfidence.HIGH
    elif confidence >= 0.4:
        level = CriticConfidence.MEDIUM
    elif confidence >= 0.2:
        level = CriticConfidence.LOW
    else:
        level = CriticConfidence.VERY_LOW

    return CriticReport(
        rule_consistency=rule_consistency,
        contradictions=contradictions or [],
        reinforcement=[],
        confidence=confidence,
        confidence_level=level,
        confidence_reason="test",
        false_positive_indicators=fp_indicators or [],
        signal_strength=signal_strength,
        evidence_sufficiency=sufficiency,
        recommended_adjustment=adjustment,
        audit_notes="test report",
        rerun_recommended=rerun,
    )


@pytest.fixture
def clean_critic():
    return _build_critic_report(confidence=0.9)


@pytest.fixture
def low_confidence_critic():
    return _build_critic_report(confidence=0.3)


@pytest.fixture
def borderline_critic():
    return _build_critic_report(
        confidence=0.75,
        sufficiency=EvidenceSufficiency.BORDERLINE,
    )


def _build_verdict(
    case_id: str = "CLEAN-001",
    rule_score: int = 2,
    rule_priority: str = "Low",
    tampered: bool = False,
    probability: float = 0.2,
    severity: Severity = Severity.Low,
    override: bool = False,
    override_reason: str = None,
    confidence: float = 0.85,
) -> ForensicVerdict:
    if confidence >= 0.9:
        level = JudgeConfidence.VERY_HIGH
    elif confidence >= 0.7:
        level = JudgeConfidence.HIGH
    elif confidence >= 0.4:
        level = JudgeConfidence.MEDIUM
    elif confidence >= 0.2:
        level = JudgeConfidence.LOW
    else:
        level = JudgeConfidence.VERY_LOW

    return ForensicVerdict(
        case_id=case_id,
        rule_based_score=rule_score,
        rule_based_priority=rule_priority,
        tampered=tampered,
        tampered_probability=probability,
        severity=severity,
        override_applied=override,
        override_reason=override_reason,
        explanation="Test explanation.",
        evidence=[
            EvidenceItem(
                source=EvidenceSource.METADATA,
                finding="test finding",
                weight=EvidenceWeight.SUPPORTING,
            )
        ],
        confidence=confidence,
        confidence_level=level,
        flagged_for_human_review=False,
    )


@pytest.fixture
def clean_verdict():
    return _build_verdict()


@pytest.fixture
def override_verdict():
    """Judge overrides High → Low (false positive)."""
    return _build_verdict(
        case_id="TAMPER-001",
        rule_score=10,
        rule_priority="High",
        tampered=False,
        probability=0.25,
        severity=Severity.Low,
        override=True,
        override_reason="Scanner workflow — false positive",
        confidence=0.85,
    )


# ═══════════════════════════════════════════════════════
# SCHEMA VALIDATION TESTS
# ═══════════════════════════════════════════════════════

class TestSchemaValidation:

    def test_priority_score_alignment_enforced(self):
        with pytest.raises(ValidationError):
            EvidenceCaseFile(
                case_id="FAIL",
                metadata=ModuleResult(score=5, signals=[]),
                font=ModuleResult(score=4, signals=[]),
                compression=ModuleResult(score=0, signals=[]),
                deterministic_score=9,
                priority=Priority.Low,
            )

    def test_high_score_requires_high_priority(self):
        with pytest.raises(ValidationError):
            EvidenceCaseFile(
                case_id="FAIL",
                metadata=ModuleResult(score=4, signals=[]),
                font=ModuleResult(score=4, signals=[]),
                compression=ModuleResult(score=2, signals=[]),
                deterministic_score=10,
                priority=Priority.Medium,
            )

    def test_valid_case_passes(self, clean_case):
        assert clean_case.case_id == "CLEAN-001"
        assert clean_case.deterministic_score == 2

    def test_verdict_preserves_rule_reference(self, clean_verdict):
        assert clean_verdict.rule_based_score == 2
        assert clean_verdict.rule_based_priority == "Low"

    def test_verdict_override_fields(self, override_verdict):
        assert override_verdict.override_applied is True
        assert override_verdict.override_reason is not None
        assert override_verdict.severity == Severity.Low
        assert override_verdict.rule_based_priority == "High"

    def test_critic_fp_fields(self):
        report = _build_critic_report(
            fp_indicators=["scanner_workflow", "single_module_dominance"],
            adjustment=AdjustmentDirection.LOWER,
        )
        assert len(report.false_positive_indicators) == 2
        assert report.recommended_adjustment == AdjustmentDirection.LOWER

    def test_critic_confidence_level_mismatch_rejected(self):
        with pytest.raises(ValidationError):
            CriticReport(
                rule_consistency=True,
                contradictions=[],
                reinforcement=[],
                confidence=0.95,
                confidence_level=CriticConfidence.LOW,
                confidence_reason="test",
                false_positive_indicators=[],
                signal_strength=SignalStrength.MODERATE,
                evidence_sufficiency=EvidenceSufficiency.SUFFICIENT,
                recommended_adjustment=AdjustmentDirection.MAINTAIN,
                audit_notes="test",
            )


# ═══════════════════════════════════════════════════════
# PREFLIGHT TESTS
# ═══════════════════════════════════════════════════════

class TestPreflight:

    def test_clean_case_no_warnings(self, clean_case):
        assert preflight_check(clean_case) is None

    def test_score_mismatch_detected(self):
        case = EvidenceCaseFile(
            case_id="MISMATCH",
            metadata=ModuleResult(score=3, signals=["a"]),
            font=ModuleResult(score=2, signals=["b"]),
            compression=ModuleResult(score=0, signals=[]),
            deterministic_score=7,
            priority=Priority.Medium,
        )
        warning = preflight_check(case)
        assert warning is not None
        assert "mismatch" in warning.lower() or "Score" in warning

    def test_high_score_no_signals_flagged(self):
        case = EvidenceCaseFile(
            case_id="NOSIG",
            metadata=ModuleResult(score=5, signals=[]),
            font=ModuleResult(score=0, signals=[]),
            compression=ModuleResult(score=0, signals=[]),
            deterministic_score=5,
            priority=Priority.Medium,
        )
        warning = preflight_check(case)
        assert warning is not None
        assert "FP RISK" in warning

    def test_single_module_dominance_flagged(self):
        case = EvidenceCaseFile(
            case_id="SINGLE",
            metadata=ModuleResult(score=7, signals=["tool_mismatch"]),
            font=ModuleResult(score=0, signals=[]),
            compression=ModuleResult(score=0, signals=[]),
            deterministic_score=7,
            priority=Priority.Medium,
        )
        warning = preflight_check(case)
        assert warning is not None
        assert "single" in warning.lower() or "one module" in warning.lower()

    def test_scanner_workflow_pattern(self):
        case = EvidenceCaseFile(
            case_id="SCAN",
            metadata=ModuleResult(score=5, signals=["multiple_tools_detected"]),
            font=ModuleResult(score=0, signals=[]),
            compression=ModuleResult(score=1, signals=["minor"]),
            deterministic_score=6,
            priority=Priority.Medium,
        )
        warning = preflight_check(case)
        assert warning is not None
        assert "scanner" in warning.lower() or "FP RISK" in warning


# ═══════════════════════════════════════════════════════
# REVIEW FLAG TESTS
# ═══════════════════════════════════════════════════════

class TestReviewFlags:

    def test_low_confidence_flags(self, clean_case):
        critic = _build_critic_report(confidence=0.3)
        result = should_flag_for_review(clean_case, critic)
        assert result is not None
        assert "confidence" in result.lower()

    def test_high_confidence_no_flag(self, clean_case):
        critic = _build_critic_report(
            confidence=0.9,
            sufficiency=EvidenceSufficiency.SUFFICIENT,
        )
        result = should_flag_for_review(clean_case, critic)
        assert result is None

    def test_borderline_evidence_flags(self, clean_case):
        critic = _build_critic_report(
            confidence=0.9,
            sufficiency=EvidenceSufficiency.BORDERLINE,
        )
        result = should_flag_for_review(clean_case, critic)
        assert result is not None
        assert "sufficiency" in result.lower() or "borderline" in result.lower()

    def test_insufficient_evidence_flags(self, clean_case):
        critic = _build_critic_report(
            confidence=0.9,
            sufficiency=EvidenceSufficiency.INSUFFICIENT,
        )
        result = should_flag_for_review(clean_case, critic)
        assert result is not None

    def test_rerun_recommended_flags(self, clean_case):
        critic = _build_critic_report(confidence=0.9, rerun=True)
        result = should_flag_for_review(clean_case, critic)
        assert result is not None
        assert "rerun" in result.lower()

    def test_high_impact_contradiction_flags(self, tampered_case):
        contradiction = Contradiction(
            contradiction_type=ContradictionType.MODULE_CONFLICT,
            modules_involved=["metadata", "compression"],
            impact=ContradictionImpact.High,
            description="Metadata says edited, compression says original",
        )
        critic = _build_critic_report(
            confidence=0.9,
            contradictions=[contradiction],
        )
        result = should_flag_for_review(tampered_case, critic)
        assert result is not None
        assert "contradiction" in result.lower()

    def test_borderline_score_flags(self):
        case = EvidenceCaseFile(
            case_id="BORDER",
            metadata=ModuleResult(score=3, signals=["a"]),
            font=ModuleResult(score=1, signals=[]),
            compression=ModuleResult(score=1, signals=[]),
            deterministic_score=5,
            priority=Priority.Medium,
        )
        critic = _build_critic_report(confidence=0.9)
        result = should_flag_for_review(case, critic)
        assert result is not None
        assert "borderline" in result.lower() or "score" in result.lower()

    def test_rule_inconsistency_flags(self, clean_case):
        critic = _build_critic_report(confidence=0.9, rule_consistency=False)
        result = should_flag_for_review(clean_case, critic)
        assert result is not None
        assert "inconsistency" in result.lower()


# ═══════════════════════════════════════════════════════
# AUDIT VERDICT TESTS
# ═══════════════════════════════════════════════════════

class TestAuditVerdict:

    def test_fixes_case_id(self, clean_case):
        verdict = _build_verdict(case_id="WRONG")
        fixed = audit_verdict(verdict, clean_case, None)
        assert fixed.case_id == "CLEAN-001"

    def test_fixes_rule_based_score(self, clean_case):
        verdict = _build_verdict(rule_score=99)
        fixed = audit_verdict(verdict, clean_case, None)
        assert fixed.rule_based_score == 2

    def test_fixes_rule_based_priority(self, clean_case):
        verdict = _build_verdict(rule_priority="High")
        fixed = audit_verdict(verdict, clean_case, None)
        assert fixed.rule_based_priority == "Low"

    def test_does_not_override_judge_severity(self, tampered_case):
        """Judge says Low even though rules say High — audit must NOT change it."""
        verdict = _build_verdict(
            case_id="TAMPER-001",
            rule_score=10,
            rule_priority="High",
            severity=Severity.Low,
            override=True,
            override_reason="False positive",
            confidence=0.85,
        )
        audited = audit_verdict(verdict, tampered_case, None)
        assert audited.severity == Severity.Low  # Judge's call preserved

    def test_does_not_override_tampered_decision(self, tampered_case):
        """Judge says not tampered even with High score — audit preserves it."""
        verdict = _build_verdict(
            case_id="TAMPER-001",
            rule_score=10,
            rule_priority="High",
            tampered=False,
            probability=0.2,
            severity=Severity.Low,
            override=True,
            override_reason="Scanner artifacts",
            confidence=0.9,
        )
        audited = audit_verdict(verdict, tampered_case, None)
        assert audited.tampered is False  # Judge's call preserved

    def test_auto_detects_override(self, tampered_case):
        """If Judge changed severity but forgot to set override_applied."""
        verdict = _build_verdict(
            case_id="TAMPER-001",
            rule_score=10,
            rule_priority="High",
            severity=Severity.Low,
            override=False,  # LLM forgot to set this
            confidence=0.85,
        )
        audited = audit_verdict(verdict, tampered_case, None)
        assert audited.override_applied is True
        assert audited.override_reason is not None

    def test_forces_review_from_precheck(self, clean_case):
        verdict = _build_verdict(confidence=0.9)
        audited = audit_verdict(verdict, clean_case, "Low critic confidence")
        assert audited.flagged_for_human_review is True
        assert "Low critic confidence" in audited.review_reason

    def test_flags_low_confidence_override(self, tampered_case):
        """Override with low confidence should be flagged."""
        verdict = _build_verdict(
            case_id="TAMPER-001",
            rule_score=10,
            rule_priority="High",
            severity=Severity.Low,
            override=True,
            override_reason="FP",
            confidence=0.6,
        )
        audited = audit_verdict(verdict, tampered_case, None)
        assert audited.flagged_for_human_review is True

    def test_high_confidence_override_not_flagged(self, tampered_case):
        """Override with high confidence should NOT be auto-flagged."""
        verdict = _build_verdict(
            case_id="TAMPER-001",
            rule_score=10,
            rule_priority="High",
            severity=Severity.Low,
            override=True,
            override_reason="Definite scanner FP",
            confidence=0.92,
        )
        audited = audit_verdict(verdict, tampered_case, None)
        assert audited.flagged_for_human_review is False


# ═══════════════════════════════════════════════════════
# SERIALIZATION TESTS
# ═══════════════════════════════════════════════════════

class TestSerialization:

    def test_serialize_excludes_none(self, clean_case):
        data = json.loads(serialize_case_file(clean_case))
        assert "qr" not in data
        assert data["case_id"] == "CLEAN-001"

    def test_serialize_includes_all_modules(self, tampered_case):
        data = json.loads(serialize_case_file(tampered_case))
        assert "metadata" in data
        assert "font" in data
        assert "compression" in data

    def test_pipeline_result_to_dict(self, clean_verdict, clean_critic):
        result = PipelineResult(
            verdict=clean_verdict,
            critic_report=clean_critic,
            case_id="X",
            duration_ms=42.5,
            success=True,
        )
        d = result.to_dict()
        assert d["success"] is True
        assert d["duration_ms"] == 42.5
        assert d["similar_cases_used"] == 0
        assert "verdict" in d
        assert "critic_report" in d

    def test_pipeline_result_error_dict(self, clean_verdict, clean_critic):
        result = PipelineResult(
            verdict=clean_verdict,
            critic_report=clean_critic,
            case_id="X",
            duration_ms=10,
            success=False,
            error="LLM timeout",
            failed_stage="judge",
        )
        d = result.to_dict()
        assert d["success"] is False
        assert d["error"] == "LLM timeout"
        assert d["failed_stage"] == "judge"


# ═══════════════════════════════════════════════════════
# PIPELINE INTEGRATION TESTS (mocked LLM)
# ═══════════════════════════════════════════════════════

class TestPipeline:

    @pytest.mark.asyncio
    async def test_full_pipeline_success(
        self, clean_case, clean_critic, clean_verdict
    ):
        critic = AsyncMock()
        critic.audit = AsyncMock(return_value=clean_critic)

        reflection = AsyncMock()
        reflection.judge = AsyncMock(return_value=clean_verdict)

        validator = ForensicValidator(critic=critic, reflection=reflection)
        result = await validator.validate(clean_case)

        assert result.success is True
        assert result.case_id == "CLEAN-001"
        assert isinstance(result.duration_ms, float)
        assert result.duration_ms > 0
        assert result.verdict.tampered is False

        critic.audit.assert_awaited_once_with(clean_case)
        reflection.judge.assert_awaited_once_with(clean_case, clean_critic)

    @pytest.mark.asyncio
    async def test_override_flows_through(
        self, tampered_case, override_verdict
    ):
        critic = AsyncMock()
        critic.audit = AsyncMock(return_value=_build_critic_report(
            fp_indicators=["scanner_workflow"],
            adjustment=AdjustmentDirection.LOWER,
        ))

        reflection = AsyncMock()
        reflection.judge = AsyncMock(return_value=override_verdict)

        validator = ForensicValidator(critic=critic, reflection=reflection)
        result = await validator.validate(tampered_case)

        assert result.success is True
        assert result.verdict.override_applied is True
        assert result.verdict.severity == Severity.Low
        assert result.verdict.rule_based_priority == "High"

    @pytest.mark.asyncio
    async def test_critic_failure_raises_pipeline_error(self, clean_case):
        critic = AsyncMock()
        critic.audit = AsyncMock(side_effect=LLMFatalError("API key invalid"))

        reflection = AsyncMock()

        validator = ForensicValidator(critic=critic, reflection=reflection)

        with pytest.raises(PipelineError) as exc_info:
            await validator.validate(clean_case)

        assert exc_info.value.stage == "critic"
        assert "API key invalid" in str(exc_info.value.cause)

    @pytest.mark.asyncio
    async def test_judge_failure_raises_pipeline_error(
        self, clean_case, clean_critic
    ):
        critic = AsyncMock()
        critic.audit = AsyncMock(return_value=clean_critic)

        reflection = AsyncMock()
        reflection.judge = AsyncMock(
            side_effect=LLMFatalError("Claude overloaded")
        )

        validator = ForensicValidator(critic=critic, reflection=reflection)

        with pytest.raises(PipelineError) as exc_info:
            await validator.validate(clean_case)

        assert exc_info.value.stage == "judge"

    @pytest.mark.asyncio
    async def test_prompt_missing_raises_pipeline_error(self, clean_case):
        critic = AsyncMock()
        critic.audit = AsyncMock(
            side_effect=FileNotFoundError("critic_prompt.txt")
        )

        reflection = AsyncMock()

        validator = ForensicValidator(critic=critic, reflection=reflection)

        with pytest.raises(PipelineError) as exc_info:
            await validator.validate(clean_case)

        assert exc_info.value.stage == "critic"

    @pytest.mark.asyncio
    async def test_storage_failure_nonblocking(
        self, clean_case, clean_critic, clean_verdict
    ):
        """Storage failures must NOT break the pipeline."""
        critic = AsyncMock()
        critic.audit = AsyncMock(return_value=clean_critic)

        reflection = AsyncMock()
        reflection.judge = AsyncMock(return_value=clean_verdict)

        validator = ForensicValidator(critic=critic, reflection=reflection)

        with patch.object(
            validator, "_store_feedback", side_effect=Exception("DB down")
        ):
            result = await validator.validate(clean_case)

        assert result.success is True
        assert result.verdict.tampered is False


# ═══════════════════════════════════════════════════════
# PROMPT LOADER TESTS
# ═══════════════════════════════════════════════════════

class TestPromptLoader:

    def test_missing_fewshots_returns_empty(self):
        with patch(
            "app.agents.critic_agent._FEWSHOT_FILE",
            MagicMock(exists=MagicMock(return_value=False)),
        ):
            PromptLoader.clear_cache()
            shots = PromptLoader.get_fewshots()
            assert shots == []
