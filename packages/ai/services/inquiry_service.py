from packages.ai.inquiry import (
    CustomerInquiry,
    InquiryCategory,
)
from packages.ai.providers.base import LLMProvider

class InquiryService:
    def __init__(self,provider: LLMProvider) -> None:
        self.provider = provider
    
    def classify(self,message: str,) -> CustomerInquiry:
        """
        Classify a customer inquiry.

        The service does not depend on a
        specific LLM vendor.
        """
        if not message.strip():
            raise ValueError(
                "Customer message cannot be empty."
            )

        inquiry = self.provider.classify_inquiry(
            message
        )
        # Deterministic business policy:
        # Billing and product issues must be
        # reviewed before executing any action.
        if inquiry.category in (
            InquiryCategory.BILLING,
            InquiryCategory.PRODUCT_ISSUE,
        ):
            inquiry = inquiry.model_copy(
                update={
                    "requires_human_approval": True
                }
            )

        return inquiry