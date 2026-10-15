import argparse
from pathlib import Path
from uuid import uuid4

from packages.ai.evaluation.dataset_runner import (
    DatasetEvaluationRunner,
)
from packages.ai.evaluation.fixture_rag import (
    FixtureRAGService,
)
from packages.ai.evaluation.quality_gate import (
    QualityGate,
    QualityThresholds,
)
from packages.ai.evaluation.runner import (
    RAGEvaluationRunner,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run AgentDesk RAG evaluation."
    )
    
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(
            "tests/ai/fixtures/rag_evaluation.json"
        ),
    )
    
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "artifacts/rag-evaluation-report.json"
        ),
    )
    
    args = parser.parse_args()

    evaluator = RAGEvaluationRunner(
        rag_service=FixtureRAGService()
    )

    runner = DatasetEvaluationRunner(
        evaluator=evaluator
    )
    
    cases = runner.load_cases(
        args.dataset
    )

    report = runner.run(
        tenant_id=uuid4(),
        cases=cases,
        dataset_name=args.dataset.stem,
    )

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    
    args.output.write_text(
        report.model_dump_json(indent=2),
        encoding="utf-8",
    )

    gate = QualityGate(
        thresholds=QualityThresholds(
            min_pass_rate=1.0,
            min_average_recall=0.8,
            min_average_precision=0.6,
        )
    )
    
    result = gate.evaluate(report)

    print(
        f"Cases: {report.summary.total_cases}"
    )

    print(
        f"Pass rate: {report.summary.pass_rate:.2%}"
    )

    print(
        "Average recall: "
        f"{report.summary.average_recall:.2%}"
    )

    print(
        "Average precision: "
        f"{report.summary.average_precision:.2%}"
    )

    print(f"Report: {args.output}")

    if not result.passed:
        print("\nQUALITY GATE FAILED")

        for failure in result.failures:
            print(f"- {failure}")

        return 1

    print("\nQUALITY GATE PASSED")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())