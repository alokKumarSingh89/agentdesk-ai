
from uuid import uuid4

from packages.ai.evaluation.models import (
    RAGEvaluationCase,
)
from packages.ai.evaluation.runner import (
    RAGEvaluationRunner,
)
from packages.ai.rag import (
    RAGResponse,
    RAGSource,
    RAGStatus,
)


class FakeRAGService:
    def __init__(self, response):
        self.response = response

    def answer(self, tenant_id, question):
        return self.response


def test_supported_answer_passes():
    response = RAGResponse(
        answer=(
            "Refunds take 5 business days."
        ),
        status=RAGStatus.ANSWERED,
        sources=[
            RAGSource(
                document_id=uuid4(),
                source="refund-policy.txt",
                content=(
                    "Refunds take 5 business days."
                ),
                similarity_score=0.9,
            )
        ],
    )

    runner = RAGEvaluationRunner(
        rag_service=FakeRAGService(response)
    )

    case = RAGEvaluationCase(
        id="refund",
        question="How long does a refund take?",
        relevant_sources=["refund-policy.txt"],
        expected_answer_terms=["5 business days"],
    )

    result = runner.evaluate(
        tenant_id=uuid4(),
        case=case,
    )

    assert result.passed is True
    assert result.retrieval.recall_at_k == 1.0


def test_incorrect_answer_fails():
    response = RAGResponse(
        answer="Refunds take 24 hours.",
        status=RAGStatus.ANSWERED,
        sources=[
            RAGSource(
                document_id=uuid4(),
                source="refund-policy.txt",
                content="Refunds take 5 business days.",
                similarity_score=0.9,
            )
        ],
    )

    runner = RAGEvaluationRunner(
        rag_service=FakeRAGService(response)
    )

    case = RAGEvaluationCase(
        id="refund",
        question="How long does a refund take?",
        relevant_sources=["refund-policy.txt"],
        expected_answer_terms=["5 business days"],
    )

    result = runner.evaluate(
        tenant_id=uuid4(),
        case=case,
    )

    assert result.passed is False
    assert result.answer_terms_present is False


def test_correct_abstention_passes():
    response = RAGResponse(
        answer=(
            "I couldn't find enough information "
            "in the available documents."
        ),
        status=RAGStatus.INSUFFICIENT_CONTEXT,
    )

    runner = RAGEvaluationRunner(
        rag_service=FakeRAGService(response)
    )

    case = RAGEvaluationCase(
        id="unknown",
        question="Who is the CEO?",
        should_abstain=True,
    )

    result = runner.evaluate(
        tenant_id=uuid4(),
        case=case,
    )

    assert result.passed is True
