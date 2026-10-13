from enum import Enum
from uuid import UUID

from pydantic import BaseModel, Field

from packages.ai.schemas import TokenUsage

class RAGStatus(str, Enum):
    ANSWERED = "answered"
    INSUFFICIENT_CONTEXT = "insufficient_context"

class RAGSource(BaseModel):
    document_id: UUID
    source: str
    content: str
    similarity_score: float
    
class RAGResponse(BaseModel):
    answer: str = Field(min_length=1)
    status: RAGStatus
    sources: list[RAGSource] = Field(
        default_factory=list
    )
    model: str | None = None

    usage: TokenUsage | None = None