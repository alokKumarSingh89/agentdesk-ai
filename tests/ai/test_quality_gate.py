
from datetime import datetime, timezone

from packages.ai.evaluation.models import (
    EvaluationReport,
    EvaluationSummary,
)
from packages.ai.evaluation.quality_gate import (
    QualityGate,
    QualityThresholds,
)


def make_report(
    pass_rate: float,
    recall: float,
    precision: float,
) -> EvaluationReport:
    return EvaluationReport(
        dataset_name="test",
        generated_at=datetime.now(timezone.utc),
        summary=EvaluationSummary(
            total_cases=10,
            passed_cases=round(pass_rate * 10),
            failed_cases=10 - round(pass_rate * 10),
            pass_rate=pass_rate,
            average_recall=recall,
            average_precision=precision,
        ),
        results=[],
    )


def test_quality_gate_passes():
    gate = QualityGate(
        QualityThresholds(
            min_pass_rate=0.9,
            min_average_recall=0.8,
            min_average_precision=0.7,
        )
    )

    report = make_report(
        pass_rate=1.0,
        recall=0.9,
        precision=0.8,
    )

    result = gate.evaluate(report)

    assert result.passed is True
    assert result.failures == []


def test_quality_gate_blocks_regression():
    gate = QualityGate(
        QualityThresholds(
            min_pass_rate=0.9,
            min_average_recall=0.8,
            min_average_precision=0.7,
        )
    )

    report = make_report(
        pass_rate=0.7,
        recall=0.6,
        precision=0.5,
    )

    result = gate.evaluate(report)

    assert result.passed is False
    assert len(result.failures) == 3
