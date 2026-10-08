from packages.ai.providers.openai_provider import (
    OpenAIProvider,
)
from packages.ai.services.semantic_search_service import (
    KnowledgeDocument,
    SemanticSearchService,
)

def main() -> None:
    provider = OpenAIProvider()

    service = SemanticSearchService(
        provider=provider
    )
    
    documents = [
         KnowledgeDocument(
            id="refund-policy",
            content=(
                "Refunds are processed within "
                "5 business days after approval."
            ),
        ),
         KnowledgeDocument(
            id="shipping-policy",
            content=(
                "Orders above ₹2,000 qualify "
                "for free shipping."
            ),
        ),
        KnowledgeDocument(
            id="return-policy",
            content=(
                "Customers can return eligible "
                "products within 30 days."
            ),
        ),
    ]
    
    print("Generating document embeddings...")
    
    service.index_documents(documents)
    
    while True:
        query = input(
            "\nCustomer question (or exit): "
        ).strip()

        if query.lower() == "exit":
            break
        
        if not query:
            continue

        results = service.search(
            query,
            top_k=3,
        )
        
        print("\nRelevant documents:\n")
        
        for result in results:
            print(
                f"{result.document.id}: "
                f"{result.score:.4f}"
            )
            print(result.document.content)
            print()
            
if __name__ == "__main__":
    main()