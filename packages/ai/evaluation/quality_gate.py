from dataclasses import dataclass

from packages.ai.evaluation.models import (
    EvaluationReport,
    QualityGateResult,
)


@dataclass(frozen=True)
class QualityThresholds:
    min_pass_rate: float = 1.0
    min_average_recall: float = 0.8
    min_average_precision: float = 0.6
    
    def __post_init__(self) -> None:
        values = (
            self.min_pass_rate,
            self.min_average_recall,
            self.min_average_precision,
        )
        
        if any(
            not 0 <= value <= 1
            for value in values
        ):
            raise ValueError(
                "Thresholds must be between 0 and 1."
            )
    
class QualityGate:
    def __init__(
        self,
        thresholds: QualityThresholds,
    ) -> None:
        self.thresholds = thresholds

    def evaluate(
        self,
        report: EvaluationReport,
    ) -> QualityGateResult:
        summary = report.summary
        
        failures: list[str] = []

        if (
            summary.pass_rate
            < self.thresholds.min_pass_rate
        ):
            failures.append(
                "Pass rate below threshold: "
                f"{summary.pass_rate:.3f} < "
                f"{self.thresholds.min_pass_rate:.3f}"
            )
        
        if (
            summary.average_recall
            < self.thresholds.min_average_recall
        ):
            failures.append(
                "Recall below threshold: "
                f"{summary.average_recall:.3f} < "
                f"{self.thresholds.min_average_recall:.3f}"
            )

        if (
            summary.average_precision
            < self.thresholds.min_average_precision
        ):
            failures.append(
                "Precision below threshold: "
                f"{summary.average_precision:.3f} < "
                f"{self.thresholds.min_average_precision:.3f}"
            )

        return QualityGateResult(
            passed=not failures,
            failures=failures,
        )