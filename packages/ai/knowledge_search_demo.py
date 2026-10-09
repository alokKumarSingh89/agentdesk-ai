from uuid import uuid4

from qdrant_client import QdrantClient

from packages.ai.knowledge import KnowledgeDocument
from packages.ai.providers.openai_provider import (
    OpenAIProvider,
)
from packages.ai.services.knowledge_search_service import (
    KnowledgeSearchService,
)
from packages.ai.vectorstores.qdrant_store import (
    QdrantVectorStore,
)
from packages.core.config import settings


def main() -> None:
    client = QdrantClient(
        url=settings.qdrant_url
    )
    
    store = QdrantVectorStore(
        client=client,
        collection_name=settings.qdrant_collection,
        vector_size=settings.embedding_dimensions,
    )
    
    store.ensure_collection()
    
    service = KnowledgeSearchService(
        provider=OpenAIProvider(),
        vector_store=store,
    )
    
    tenant_a = uuid4()
    tenant_b = uuid4()

    print(f"Tenant A: {tenant_a}")
    print(f"Tenant B: {tenant_b}")
    
    documents = [
        KnowledgeDocument(
            id=uuid4(),
            tenant_id=tenant_a,
            source="refund-policy",
            content=(
                "Refunds are processed within "
                "5 business days after approval."
            ),
        ),
        KnowledgeDocument(
            id=uuid4(),
            tenant_id=tenant_a,
            source="shipping-policy",
            content=(
                "Orders above ₹2,000 qualify "
                "for free delivery."
            ),
        ),
        KnowledgeDocument(
            id=uuid4(),
            tenant_id=tenant_b,
            source="private-policy",
            content=(
                "Tenant B provides a special "
                "90-day warranty."
            ),
        ),
    ]
    
    for document in documents:
        service.index_document(document)
    
    print("\nDocuments indexed.")

    while True:
        query = input(
            "\nTenant A question (or exit): "
        ).strip()
        
        if query.lower() == "exit":
            break

        if not query:
            continue

        results = service.search(
            tenant_id=tenant_a,
            query=query,
            top_k=3,
        )
        
        for result in results:
            print(
                f"\nScore: {result.score:.4f}"
            )
            print(
                f"Source: {result.document.source}"
            )
            print(result.document.content)

if __name__ == "__main__":
    main()