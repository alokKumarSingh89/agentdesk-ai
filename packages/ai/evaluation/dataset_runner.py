
import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID

from pydantic import TypeAdapter

from packages.ai.evaluation.models import (
    EvaluationReport,
    EvaluationSummary,
    RAGEvaluationCase,
)
from packages.ai.evaluation.runner import (
    RAGEvaluationRunner,
)


class DatasetEvaluationRunner:
    def __init__(
        self,
        evaluator: RAGEvaluationRunner,
    ) -> None:
        self.evaluator = evaluator

    @staticmethod
    def load_cases(
        dataset_path: Path,
    ) -> list[RAGEvaluationCase]:
        raw = json.loads(
            dataset_path.read_text(
                encoding="utf-8"
            )
        )

        adapter = TypeAdapter(
            list[RAGEvaluationCase]
        )

        cases = adapter.validate_python(raw)

        if not cases:
            raise ValueError(
                "Evaluation dataset cannot be empty."
            )

        ids = [case.id for case in cases]

        if len(ids) != len(set(ids)):
            raise ValueError(
                "Evaluation case IDs must be unique."
            )

        return cases

    def run(
        self,
        tenant_id: UUID,
        cases: list[RAGEvaluationCase],
        dataset_name: str,
    ) -> EvaluationReport:
        if not cases:
            raise ValueError(
                "Evaluation dataset cannot be empty."
            )

        results = [
            self.evaluator.evaluate(
                tenant_id=tenant_id,
                case=case,
            )
            for case in cases
        ]

        total = len(results)

        passed = sum(
            result.passed
            for result in results
        )

        summary = EvaluationSummary(
            total_cases=total,
            passed_cases=passed,
            failed_cases=total - passed,
            pass_rate=passed / total,
            average_recall=sum(
                result.retrieval.recall_at_k
                for result in results
            ) / total,
            average_precision=sum(
                result.retrieval.precision_at_k
                for result in results
            ) / total,
        )

        return EvaluationReport(
            dataset_name=dataset_name,
            generated_at=datetime.now(timezone.utc),
            summary=summary,
            results=results,
        )
