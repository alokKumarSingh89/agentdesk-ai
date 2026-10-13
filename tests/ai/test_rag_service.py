
from decimal import Decimal
from uuid import uuid4

import pytest

from packages.ai.knowledge import (
    KnowledgeDocument,
    KnowledgeSearchResult,
)
from packages.ai.rag import RAGStatus
from packages.ai.schemas import (
    LLMResponse,
    TokenUsage,
)
from packages.ai.services.rag_service import RAGService


class FakeRetriever:
    def __init__(self, results=None):
        self.results = results or []
        self.received_tenant_id = None

    def search(
        self,
        tenant_id,
        query,
        top_k=3,
    ):
        self.received_tenant_id = tenant_id
        return self.results[:top_k]


class FakeGenerator:
    def __init__(self):
        self.calls = 0
        self.received_prompt = None

    def generate(self, prompt):
        self.calls += 1
        self.received_prompt = prompt

        return LLMResponse(
            content=(
                "Refunds take 5 business days "
                "after approval. [1]"
            ),
            model="fake-model",
            usage=TokenUsage(
                input_tokens=20,
                output_tokens=10,
                total_tokens=30,
                estimated_cost_usd=Decimal("0"),
            ),
        )


def make_result(
    tenant_id,
    score=0.9,
):
    return KnowledgeSearchResult(
        document=KnowledgeDocument(
            id=uuid4(),
            tenant_id=tenant_id,
            source="refund-policy.txt",
            content=(
                "Refunds are processed within "
                "5 business days after approval."
            ),
        ),
        score=score,
    )


def test_rag_answers_with_retrieved_context():
    tenant_id = uuid4()

    retriever = FakeRetriever(
        results=[
            make_result(tenant_id, score=0.9)
        ]
    )

    generator = FakeGenerator()

    service = RAGService(
        retriever=retriever,
        generator=generator,
    )

    response = service.answer(
        tenant_id=tenant_id,
        question="How long does a refund take?",
    )

    assert response.status == RAGStatus.ANSWERED
    assert len(response.sources) == 1
    assert response.sources[0].source == (
        "refund-policy.txt"
    )

    assert "5 business days" in response.answer

    assert "refund-policy.txt" in (
        generator.received_prompt
    )

    assert generator.calls == 1


def test_no_results_does_not_call_llm():
    retriever = FakeRetriever()
    generator = FakeGenerator()

    service = RAGService(
        retriever=retriever,
        generator=generator,
    )

    response = service.answer(
        tenant_id=uuid4(),
        question="What is your warranty?",
    )

    assert (
        response.status
        == RAGStatus.INSUFFICIENT_CONTEXT
    )

    assert response.sources == []
    assert generator.calls == 0


def test_low_similarity_results_are_rejected():
    tenant_id = uuid4()

    retriever = FakeRetriever(
        results=[
            make_result(
                tenant_id,
                score=0.2,
            )
        ]
    )

    generator = FakeGenerator()

    service = RAGService(
        retriever=retriever,
        generator=generator,
        min_similarity=0.5,
    )

    response = service.answer(
        tenant_id=tenant_id,
        question="Who is the CEO?",
    )

    assert (
        response.status
        == RAGStatus.INSUFFICIENT_CONTEXT
    )

    assert generator.calls == 0


def test_tenant_id_is_forwarded():
    tenant_id = uuid4()

    retriever = FakeRetriever()
    generator = FakeGenerator()

    service = RAGService(
        retriever=retriever,
        generator=generator,
    )

    service.answer(
        tenant_id=tenant_id,
        question="Refund policy?",
    )

    assert (
        retriever.received_tenant_id
        == tenant_id
    )


def test_empty_question_is_rejected():
    retriever = FakeRetriever()
    generator = FakeGenerator()

    service = RAGService(
        retriever=retriever,
        generator=generator,
    )

    with pytest.raises(ValueError):
        service.answer(
            tenant_id=uuid4(),
            question="   ",
        )

    assert generator.calls == 0


def test_invalid_similarity_threshold():
    with pytest.raises(ValueError):
        RAGService(
            retriever=FakeRetriever(),
            generator=FakeGenerator(),
            min_similarity=1.5,
        )
