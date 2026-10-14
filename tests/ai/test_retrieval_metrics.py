
import pytest

from packages.ai.evaluation.metrics import (
    calculate_retrieval_metrics,
)


def test_perfect_retrieval():
    metrics = calculate_retrieval_metrics(
        retrieved_sources=["a", "b"],
        relevant_sources=["a", "b"],
    )

    assert metrics.precision_at_k == 1.0
    assert metrics.recall_at_k == 1.0


def test_partial_retrieval():
    metrics = calculate_retrieval_metrics(
        retrieved_sources=["a", "c"],
        relevant_sources=["a", "b"],
    )

    assert metrics.precision_at_k == pytest.approx(0.5)
    assert metrics.recall_at_k == pytest.approx(0.5)


def test_no_retrieval():
    metrics = calculate_retrieval_metrics(
        retrieved_sources=[],
        relevant_sources=["a"],
    )

    assert metrics.precision_at_k == 0.0
    assert metrics.recall_at_k == 0.0


def test_no_relevant_documents():
    metrics = calculate_retrieval_metrics(
        retrieved_sources=[],
        relevant_sources=[],
    )

    assert metrics.precision_at_k == 0.0
    assert metrics.recall_at_k == 1.0
