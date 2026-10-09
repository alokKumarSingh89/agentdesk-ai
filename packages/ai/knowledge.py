from uuid import UUID

from pydantic import BaseModel, Field

class KnowledgeDocument(BaseModel):
    id: UUID
    tenant_id: UUID
    content: str = Field(min_length=1)
    source: str = Field(min_length=1)
    
class KnowledgeSearchResult(BaseModel):
    document: KnowledgeDocument
    score: float