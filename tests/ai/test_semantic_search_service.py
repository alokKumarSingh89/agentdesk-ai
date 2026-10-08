
import pytest

from packages.ai.embeddings import EmbeddingResult
from packages.ai.services.semantic_search_service import (
    KnowledgeDocument,
    SemanticSearchService,
)


class FakeEmbeddingProvider:
    def __init__(self) -> None:
        self.vectors = {
            "Refunds are processed in 5 days": [
                1.0, 0.0
            ],
            "Shipping takes 3 days": [
                0.0, 1.0
            ],
            "When will I get my refund?": [
                0.9, 0.1
            ],
        }

    def embed(
        self,
        text: str,
    ) -> EmbeddingResult:
        return EmbeddingResult(
            model="fake-embedding-model",
            vector=self.vectors[text],
            token_usage=0,
        )


def test_search_returns_most_relevant_document():
    service = SemanticSearchService(
        provider=FakeEmbeddingProvider()
    )

    service.index_documents([
        KnowledgeDocument(
            id="refund",
            content="Refunds are processed in 5 days",
        ),
        KnowledgeDocument(
            id="shipping",
            content="Shipping takes 3 days",
        ),
    ])

    results = service.search(
        "When will I get my refund?",
        top_k=1,
    )

    assert len(results) == 1
    assert results[0].document.id == "refund"


def test_empty_query_is_rejected():
    service = SemanticSearchService(
        provider=FakeEmbeddingProvider()
    )

    with pytest.raises(ValueError):
        service.search("   ")


def test_search_before_indexing_returns_empty():
    service = SemanticSearchService(
        provider=FakeEmbeddingProvider()
    )

    assert service.search(
        "When will I get my refund?"
    ) == []
