from pathlib import Path
from uuid import UUID, uuid5

from packages.ai.ingestion.chunker import TextChunker
from packages.ai.ingestion.extractor import (
    DocumentTextExtractor,
)
from packages.ai.knowledge import KnowledgeDocument
from packages.ai.services.knowledge_search_service import (
    KnowledgeSearchService,
)


class DocumentIngestionService:
    def __init__(
        self,
        extractor: DocumentTextExtractor,
        chunker: TextChunker,
        knowledge_service: KnowledgeSearchService,
    ) -> None:
        self.extractor = extractor
        self.chunker = chunker
        self.knowledge_service = knowledge_service
    
    def ingest(
        self,
        tenant_id: UUID,
        document_id: UUID,
        file_path: Path,
    ) -> int:
        text = self.extractor.extract(file_path)
        chunks = self.chunker.split(text)
        if not chunks:
            raise ValueError(
                "No chunks were generated."
            )
            
        for index, chunk in enumerate(chunks):
            chunk_id = uuid5(
                document_id,
                f"chunk:{index}",
            )
            document = KnowledgeDocument(
                    id=chunk_id,
                    tenant_id=tenant_id,
                    content=chunk,
                    source=file_path.name,
                )
            self.knowledge_service.index_document(
                document
            )
        return len(chunks)