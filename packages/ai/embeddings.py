from pydantic import BaseModel, Field

class EmbeddingResult(BaseModel):
    model: str
    vector: list[float] = Field(min_length=1)
    token_usage: int = Field(ge=0)