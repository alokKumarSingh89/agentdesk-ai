"""Pydantic schemas for LLM responses."""
from decimal import Decimal

from pydantic import BaseModel, Field

class TokenUsage(BaseModel):
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    total_tokens: int = Field(ge=0)

    estimated_cost_usd: Decimal | None = None
    

class LLMResponse(BaseModel):
    content: str
    model: str
    usage: TokenUsage