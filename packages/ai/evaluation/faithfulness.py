
from pydantic import BaseModel, Field


class FaithfulnessJudgment(BaseModel):
    faithful: bool

    explanation: str = Field(
        min_length=1
    )

    unsupported_claims: list[str] = Field(
        default_factory=list
    )
