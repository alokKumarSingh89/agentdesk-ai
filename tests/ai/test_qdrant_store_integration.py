
import os
from uuid import uuid4

import pytest
from qdrant_client import QdrantClient

from packages.ai.knowledge import KnowledgeDocument
from packages.ai.vectorstores.qdrant_store import (
    QdrantVectorStore,
)


@pytest.fixture
def store():
    url = os.getenv(
        "QDRANT_URL",
        "http://localhost:6333",
    )

    client = QdrantClient(url=url, timeout=5)

    collection = f"test_agentdesk_{uuid4().hex}"

    vector_store = QdrantVectorStore(
        client=client,
        collection_name=collection,
        vector_size=3,
    )

    vector_store.ensure_collection()

    try:
        yield vector_store
    finally:
        client.delete_collection(collection)


def test_tenant_isolation(store):
    tenant_a = uuid4()
    tenant_b = uuid4()

    document_a = KnowledgeDocument(
        id=uuid4(),
        tenant_id=tenant_a,
        content="Tenant A refund policy",
        source="refund-policy",
    )

    document_b = KnowledgeDocument(
        id=uuid4(),
        tenant_id=tenant_b,
        content="Tenant B private policy",
        source="private-policy",
    )

    store.upsert(
        document_a,
        [1.0, 0.0, 0.0],
    )

    store.upsert(
        document_b,
        [0.99, 0.01, 0.0],
    )

    results = store.search(
        tenant_id=tenant_a,
        vector=[1.0, 0.0, 0.0],
        top_k=10,
    )

    assert len(results) == 1

    assert results[0].document.id == document_a.id
    assert results[0].document.tenant_id == tenant_a


def test_collection_persists_vectors(store):
    tenant_id = uuid4()

    document = KnowledgeDocument(
        id=uuid4(),
        tenant_id=tenant_id,
        content="Return policy",
        source="return-policy",
    )

    store.upsert(
        document,
        [0.0, 1.0, 0.0],
    )

    results = store.search(
        tenant_id=tenant_id,
        vector=[0.0, 1.0, 0.0],
    )

    assert len(results) == 1
    assert results[0].document.content == "Return policy"


def test_wrong_vector_dimensions_rejected(store):
    tenant_id = uuid4()

    with pytest.raises(ValueError):
        store.search(
            tenant_id=tenant_id,
            vector=[1.0, 0.0],
        )
