from typing import Protocol
from uuid import UUID

from packages.ai.embeddings import EmbeddingResult
from packages.ai.knowledge import (
    KnowledgeDocument,
    KnowledgeSearchResult,
)
from packages.ai.vectorstores.qdrant_store import (
    QdrantVectorStore,
)

class EmbeddingProvider(Protocol):
    def embed(
        self,
        text: str,
    ) -> EmbeddingResult:
        ...

class KnowledgeSearchService:

    def __init__(
        self,
        provider: EmbeddingProvider,
        vector_store: QdrantVectorStore,
    ) -> None:
        self.provider = provider
        self.vector_store = vector_store
    
    def index_document(
        self,
        document: KnowledgeDocument,
    ) -> None:
        result = self.provider.embed(
            document.content
        )
        self.vector_store.upsert(
            document=document,
            vector=result.vector,
        )
    
    def search(
        self,
        tenant_id: UUID,
        query: str,
        top_k: int = 3,
    ) -> list[KnowledgeSearchResult]:
        query = query.strip()
        
        if not query:
            raise ValueError(
                "Search query cannot be empty."
            )
            
        if top_k < 1:
            raise ValueError(
                "top_k must be positive."
            )
        
        result = self.provider.embed(query)
        
        return self.vector_store.search(
            tenant_id=tenant_id,
            vector=result.vector,
            top_k=top_k,
        )