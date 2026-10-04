from enum import Enum

from pydantic import BaseModel, ConfigDict, Field

class InquiryCategory(str, Enum):
    PRODUCT_ISSUE = "product_issue"
    BILLING = "billing"
    SALES = "sales"
    GENERAL = "general"
    OTHER = "other"
    
class InquiryUrgency(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    
class CustomerInquiry(BaseModel):
    model_config = ConfigDict(extra="forbid")
    
    intent: str = Field(
        description="Short snake_case customer intent."
    )
    category: InquiryCategory
    urgency: InquiryUrgency
    summary: str = Field(
        description="A concise summary of the inquiry."
    )
    suggested_action: str = Field(
        description="Recommended next step for the business."
    )
    requires_human_approval: bool = Field(
        description=(
            "Whether the suggested action requires "
            "a human decision before execution."
        )
    )