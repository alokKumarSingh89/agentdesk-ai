
from decimal import Decimal

from pydantic import Field
from pydantic_settings import (
    BaseSettings,
    SettingsConfigDict,
)


class Settings(BaseSettings):
    openai_api_key: str

    openai_model: str = "gpt-5-mini"
    
    embedding_model: str = "text-embedding-3-small"
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "agentdesk_knowledge"
    embedding_dimensions: int = 1536

    # USD per million tokens.
    # Optional until pricing is configured.
    input_price_per_million: Decimal | None = Field(
        default=None,
        ge=0,
    )

    output_price_per_million: Decimal | None = Field(
        default=None,
        ge=0,
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_ignore_empty=True,
        extra="ignore",
    )


settings = Settings()
