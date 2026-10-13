import argparse
from uuid import UUID

from qdrant_client import QdrantClient

from packages.ai.providers.openai_provider import (
    OpenAIProvider,
)
from packages.ai.services.knowledge_search_service import (
    KnowledgeSearchService,
)
from packages.ai.services.rag_service import RAGService
from packages.ai.vectorstores.qdrant_store import (
    QdrantVectorStore,
)
from packages.core.config import settings

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ask AgentDesk AI using RAG."
    )
    
    parser.add_argument(
        "--tenant-id",
        type=UUID,
        required=True,
    )
    args = parser.parse_args()

    provider = OpenAIProvider()

    client = QdrantClient(
        url=settings.qdrant_url
    )
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=settings.qdrant_collection,
        vector_size=settings.embedding_dimensions,
    )

    vector_store.ensure_collection()
    
    knowledge_service = KnowledgeSearchService(
        provider=provider,
        vector_store=vector_store,
    )

    rag_service = RAGService(
        retriever=knowledge_service,
        generator=provider,
        top_k=3,
        min_similarity=0.5,
    )

    print("\nAgentDesk AI — RAG Assistant")
    print("Type 'exit' to quit.")
    
    while True:
        question = input(
            "\nCustomer question: "
        ).strip()

        if question.lower() == "exit":
            break

        if not question:
            continue
        
        try:
            response = rag_service.answer(
                tenant_id=args.tenant_id,
                question=question,
            )
            print("\nAgentDesk:")
            print(response.answer)

            if response.sources:
                print("\nRetrieved sources:")

                for index, source in enumerate(
                    response.sources,
                    start=1,
                ):
                    print(
                        f"[{index}] {source.source} "
                        f"(score={source.similarity_score:.3f})"
                    )

            print(
                f"\nStatus: {response.status.value}"
            )

            if response.usage:
                print(
                    "Total tokens: "
                    f"{response.usage.total_tokens}"
                )

        except Exception as exc:
            print(
                f"\nRequest failed: "
                f"{type(exc).__name__}: {exc}"
            )

if __name__ == "__main__":
    main()