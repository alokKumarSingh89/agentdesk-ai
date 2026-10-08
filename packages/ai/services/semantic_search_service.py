from dataclasses import dataclass

from packages.ai.providers.base import LLMProvider
from packages.ai.similarity import cosine_similarity

@dataclass(frozen=True)
class KnowledgeDocument:
    id: str
    content: str
    

@dataclass(frozen=True)
class SearchResult:
    document: KnowledgeDocument
    score: float

class SemanticSearchService:
    def __init__(self,provider: LLMProvider) -> None:
        self.provider = provider
        self._documents: list[
            tuple[KnowledgeDocument, list[float]]
        ] = []
    
    def index_documents(
        self,
        documents: list[KnowledgeDocument],
    ) -> None:
        indexed = []
        
        for document in documents:
            result = self.provider.embed(
                document.content
            )
            indexed.append(
                (document, result.vector)
            )
        
        # Replace the index only after all
        # embeddings have been generated.
        self._documents = indexed
    
    def search(
        self,
        query: str,
        top_k: int = 3,
    ) -> list[SearchResult]:
        if not query.strip():
            raise ValueError(
                "Search query cannot be empty."
            )

        if top_k < 1:
            raise ValueError(
                "top_k must be positive."
            )
        if not self._documents:
            return []
        
        query_vector = self.provider.embed(query).vector
        results = [
            SearchResult(
                document=document,
                score=cosine_similarity(
                    query_vector,
                    document_vector,
                ),
            )
            
            for document, document_vector in self._documents
        ]
        
        results.sort(
            key=lambda item: item.score,
            reverse=True,
        )
        
        return results[:top_k]