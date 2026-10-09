from uuid import UUID
from qdrant_client import QdrantClient, models

from packages.ai.knowledge import (
    KnowledgeDocument,
    KnowledgeSearchResult,
)

class QdrantVectorStore:

    def __init__(
        self,
        client: QdrantClient,
        collection_name: str,
        vector_size: int,
    ) -> None:
        if vector_size < 1:
            raise ValueError(
                "Vector size must be positive."
            )
        
        self.client = client
        self.collection_name = collection_name
        self.vector_size = vector_size
    
    def ensure_collection(self) -> None:
        if self.client.collection_exists(
            self.collection_name
        ):
            collection = self.client.get_collection(self.collection_name)
            vectors = collection.config.params.vectors
            
            if not isinstance(vectors, models.VectorParams):
                raise RuntimeError(
                    "Expected a single unnamed vector."
                )
            
            if vectors.size != self.vector_size:
                raise RuntimeError(
                    "Existing collection dimensions "
                    "do not match configuration."
                )
            
            if vectors.distance != models.Distance.COSINE:
                raise RuntimeError(
                    "Existing collection distance "
                    "must be cosine."
                )
            
            return 
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(
                size=self.vector_size,
                distance=models.Distance.COSINE
            )
        )
        
        self.client.create_payload_index(
            collection_name=self.collection_name,
            field_name="tenant_id",
            field_schema=models.PayloadSchemaType.KEYWORD,
            wait=True
        )
    
    def upsert(self,document: KnowledgeDocument,vector: list[float]) -> None:
        if len(vector) != self.vector_size:
            raise ValueError(
                "Embedding dimensions do not match "
                "the Qdrant collection."
            )
        self.client.upsert(
            collection_name=self.collection_name,
            points=[
                models.PointStruct(
                    id=str(document.id),
                    vector=vector,
                    payload={
                        "tenant_id": str(
                            document.tenant_id
                        ),
                        "content": document.content,
                        "source": document.source,
                    }
                )
            ],
            wait=True
        )
    
    def search(
        self,
        tenant_id: UUID,
        vector: list[float],
        top_k: int = 3,
    ) -> list[KnowledgeSearchResult]:
        if top_k < 1:
            raise ValueError(
                "top_k must be positive."
            )

        if len(vector) != self.vector_size:
            raise ValueError(
                "Query embedding dimensions "
                "do not match the collection."
            )
        
        response = self.client.query_points(
            collection_name=self.collection_name,
            query=vector,
            query_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="tenant_id",
                        match=models.MatchValue(value=str(tenant_id))
                    )
                ]
            ),
            limit=top_k,
            with_payload=True
        )
        
        results = []

        for point in response.points:
            payload = point.payload or {}
            document = KnowledgeDocument(
                id=UUID(str(point.id)),
                tenant_id=UUID(payload["tenant_id"]),
                content=payload["content"],
                source=payload["source"],
            )
            results.append(
                KnowledgeSearchResult(
                    document=document,
                    score=point.score,
                )
            )
        return results

            