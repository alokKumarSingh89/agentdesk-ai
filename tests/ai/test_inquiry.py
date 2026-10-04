import pytest
from pydantic import ValidationError

from packages.ai.inquiry import (
    CustomerInquiry,
    InquiryCategory,
    InquiryUrgency,
)

def test_valid_customer_inquiry() -> None:
    inquiry = CustomerInquiry(
        intent="replacement_request",
        category=InquiryCategory.PRODUCT_ISSUE,
        urgency=InquiryUrgency.HIGH,
        summary="Customer requests replacement.",
        suggested_action="Verify eligibility.",
        requires_human_approval=True,
    )

    assert inquiry.urgency == InquiryUrgency.HIGH
    assert inquiry.requires_human_approval is True

def test_invalid_urgency_is_rejected() -> None:
    with pytest.raises(ValidationError):
        CustomerInquiry.model_validate(
            {
                "intent": "refund_request",
                "category": "billing",
                "urgency": "CRITICAL",
                "summary": "Customer requests refund.",
                "suggested_action": "Review request.",
                "requires_human_approval": True,
            }
        )
def test_extra_fields_are_rejected() -> None:
    with pytest.raises(ValidationError):
        CustomerInquiry.model_validate(
            {
                "intent": "sales_inquiry",
                "category": "sales",
                "urgency": "LOW",
                "summary": "Customer asks about pricing.",
                "suggested_action": "Share product details.",
                "requires_human_approval": False,
                "unexpected_field": "not allowed",
            }
        )