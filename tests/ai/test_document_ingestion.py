
from uuid import uuid4

from packages.ai.ingestion.chunker import TextChunker
from packages.ai.ingestion.extractor import (
    DocumentTextExtractor,
)
from packages.ai.ingestion.service import (
    DocumentIngestionService,
)


class FakeKnowledgeService:
    def __init__(self):
        self.documents = []

    def index_document(self, document):
        self.documents.append(document)


def test_document_is_chunked_and_indexed(tmp_path):
    file_path = tmp_path / "policy.txt"

    file_path.write_text(
        "Refund policy. " * 50,
        encoding="utf-8",
    )

    knowledge_service = FakeKnowledgeService()

    service = DocumentIngestionService(
        extractor=DocumentTextExtractor(),
        chunker=TextChunker(
            chunk_size=20,
            chunk_overlap=5,
        ),
        knowledge_service=knowledge_service,
    )

    tenant_id = uuid4()
    document_id = uuid4()

    count = service.ingest(
        tenant_id=tenant_id,
        document_id=document_id,
        file_path=file_path,
    )

    assert count > 1
    assert len(knowledge_service.documents) == count

    assert all(
        document.tenant_id == tenant_id
        for document in knowledge_service.documents
    )


def test_reingestion_uses_stable_chunk_ids(tmp_path):
    file_path = tmp_path / "policy.txt"

    file_path.write_text(
        "Refund policy. " * 20,
        encoding="utf-8",
    )

    knowledge_service = FakeKnowledgeService()

    service = DocumentIngestionService(
        extractor=DocumentTextExtractor(),
        chunker=TextChunker(
            chunk_size=20,
            chunk_overlap=5,
        ),
        knowledge_service=knowledge_service,
    )

    tenant_id = uuid4()
    document_id = uuid4()

    service.ingest(
        tenant_id,
        document_id,
        file_path,
    )

    first_ids = [
        document.id
        for document in knowledge_service.documents
    ]

    knowledge_service.documents.clear()

    service.ingest(
        tenant_id,
        document_id,
        file_path,
    )

    second_ids = [
        document.id
        for document in knowledge_service.documents
    ]

    assert first_ids == second_ids
