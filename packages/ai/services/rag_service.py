from typing import Protocol
from uuid import UUID

from packages.ai.knowledge import KnowledgeSearchResult
from packages.ai.prompts.rag import RAG_PROMPT_V1
from packages.ai.rag import (
    RAGResponse,
    RAGSource,
    RAGStatus,
)
from packages.ai.schemas import LLMResponse

class KnowledgeRetriever(Protocol):
    def search(
        self,
        tenant_id: UUID,
        query: str,
        top_k: int = 3,
    ) -> list[KnowledgeSearchResult]:
        ...
        
class TextGenerator(Protocol):
    def generate(
        self,
        prompt: str,
    ) -> LLMResponse:
        ...
        
class RAGService:
    INSUFFICIENT_CONTEXT_MESSAGE = (
        "I couldn't find enough information in the available documents to answer your question."
    )
    
    def __init__(
        self,
        retriever: KnowledgeRetriever,
        generator: TextGenerator,
        top_k: int = 3,
        min_similarity: float = 0.5,
    ) -> None:
        if top_k < 1:
            raise ValueError(
                "top_k must be positive."
            )

        if not -1.0 <= min_similarity <= 1.0:
            raise ValueError(
                "Similarity threshold must be "
                "between -1 and 1."
            )
        self.retriever = retriever
        self.generator = generator
        self.top_k = top_k
        self.min_similarity = min_similarity
    
    def answer(
        self,
        tenant_id: UUID,
        question: str,
    ) -> RAGResponse:
        question = question.strip()
        if not question:
            raise ValueError(
                "Question cannot be empty."
            )
        results = self.retriever.search(
            tenant_id=tenant_id,
            query=question,
            top_k=self.top_k,
        )
        
        relevant_results = [
            result
            for result in results
            if result.score >= self.min_similarity
        ]
        
        if not relevant_results:
            return RAGResponse(
                answer=self.INSUFFICIENT_CONTEXT_MESSAGE,
                status=RAGStatus.INSUFFICIENT_CONTEXT,
            )
        
        sources = [
            RAGSource(
                document_id=result.document.id,
                source=result.document.source,
                content=result.document.content,
                similarity_score=result.score,
            )
            for result in relevant_results
        ]
        context = self._build_context(sources)
        
        prompt = RAG_PROMPT_V1.render(
            question=question,
            context=context,
        )
        
        llm_response = self.generator.generate(
            prompt
        )
        answer = llm_response.content.strip()
        
        if not answer:
            raise RuntimeError(
                "LLM returned an empty RAG answer."
            )
            
        return RAGResponse(
            answer=answer,
            status=RAGStatus.ANSWERED,
            sources=sources,
            model=llm_response.model,
            usage=llm_response.usage,
        )
    
    @staticmethod
    def _build_context(
        sources: list[RAGSource],
    ) -> str:
        sections = []
        
        for index, source in enumerate(
            sources,
            start=1,
        ):
            sections.append(
                f"[{index}]\n"
                f"Source: {source.source}\n"
                f"Content:\n{source.content}"
            )
            
        return "\n\n".join(sections)