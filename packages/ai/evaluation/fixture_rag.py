from uuid import UUID, uuid5

from packages.ai.rag import (
    RAGResponse,
    RAGSource,
    RAGStatus,
)


class FixtureRAGService:
    NAMESPACE = UUID(
        "b43d6218-90d0-4e5d-9f2d-754873a1f2c3"
    )
    
    def answer(
        self,
        tenant_id: UUID,
        question: str,
    ) -> RAGResponse:
        if question == (
            "How long does a refund take after approval?"
        ):
            return self._answered(
                question=question,
                answer=(
                    "Refunds take 5 business days "
                    "after approval. [1]"
                ),
                content=(
                    "Refunds are processed within "
                    "5 business days after approval."
                ),
            )
        
        if question == (
            "What should I do if my product arrives damaged?"
        ):
            return self._answered(
                question=question,
                answer=(
                    "Contact support with your "
                    "order number and a photograph "
                    "of the damaged item. [1]"
                ),
                content=(
                    "For damaged products, contact "
                    "support with your order number "
                    "and a photograph."
                ),
            )
        if question == (
            "Who is the CEO of this company?"
        ):
            return RAGResponse(
                answer=(
                    "I couldn't find enough information "
                    "in the available documents."
                ),
                status=RAGStatus.INSUFFICIENT_CONTEXT,
            )

        raise ValueError(
            f"No fixture for question: {question}"
        )
        
    @classmethod
    def _answered(
        cls,
        question: str,
        answer: str,
        content: str,
    ) -> RAGResponse:
        return RAGResponse(
            answer=answer,
            status=RAGStatus.ANSWERED,
            sources=[
                RAGSource(
                    document_id=uuid5(
                        cls.NAMESPACE,
                        question,
                    ),
                    source="refund-policy.txt",
                    content=content,
                    similarity_score=0.95,
                )
            ],
        )