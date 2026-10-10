import argparse
from pathlib import Path
from uuid import UUID

from qdrant_client import QdrantClient

from packages.ai.ingestion.chunker import TextChunker
from packages.ai.ingestion.extractor import (
    DocumentTextExtractor,
)
from packages.ai.ingestion.service import (
    DocumentIngestionService,
)
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
    parser = argparse.ArgumentParser(description="Ingest a document into AgentDesk AI.")
    parser.add_argument(
        "--file",
        type=Path,
        required=True
    )
    
    parser.add_argument(
        "--tenant-id",
        type=UUID,
        required=True
    )
    
    parser.add_argument(
        "--document-id",
        type=UUID,
        required=True,
    )
    args = parser.parse_args()
    
    client = QdrantClient(
        url=settings.qdrant_url
    )

    store = QdrantVectorStore(
        client=client,
        collection_name=settings.qdrant_collection,
        vector_size=settings.embedding_dimensions,
    )

    store.ensure_collection()
    
    knowledge_service = KnowledgeSearchService(
        provider=OpenAIProvider(),
        vector_store=store,
    )

    ingestion_service = DocumentIngestionService(
        extractor=DocumentTextExtractor(),
        chunker=TextChunker(
            chunk_size=300,
            chunk_overlap=50,
        ),
        knowledge_service=knowledge_service,
    )

    chunk_count = ingestion_service.ingest(
        tenant_id=args.tenant_id,
        document_id=args.document_id,
        file_path=args.file,
    )

    print(
        f"Successfully indexed {chunk_count} chunks."
    )

if __name__ == "__main__":
    main()