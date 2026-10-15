
import json
from uuid import uuid4

import pytest

from packages.ai.evaluation.dataset_runner import (
    DatasetEvaluationRunner,
)
from packages.ai.evaluation.fixture_rag import (
    FixtureRAGService,
)
from packages.ai.evaluation.runner import (
    RAGEvaluationRunner,
)


def make_runner():
    return DatasetEvaluationRunner(
        evaluator=RAGEvaluationRunner(
            rag_service=FixtureRAGService()
        )
    )


def test_dataset_runner_passes_golden_cases():
    runner = make_runner()

    cases = runner.load_cases(
        __import__("pathlib").Path(
            "tests/ai/fixtures/rag_evaluation.json"
        )
    )

    report = runner.run(
        tenant_id=uuid4(),
        cases=cases,
        dataset_name="golden",
    )

    assert report.summary.total_cases == 3
    assert report.summary.passed_cases == 3
    assert report.summary.failed_cases == 0
    assert report.summary.pass_rate == 1.0


def test_empty_dataset_rejected(tmp_path):
    dataset = tmp_path / "empty.json"
    dataset.write_text("[]", encoding="utf-8")

    with pytest.raises(ValueError):
        DatasetEvaluationRunner.load_cases(
            dataset
        )


def test_duplicate_case_ids_rejected(tmp_path):
    case = {
        "id": "duplicate",
        "question": "Example question"
    }

    dataset = tmp_path / "duplicate.json"

    dataset.write_text(
        json.dumps([case, case]),
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        DatasetEvaluationRunner.load_cases(
            dataset
        )
