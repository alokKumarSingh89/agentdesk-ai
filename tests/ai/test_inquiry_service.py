
from decimal import Decimal

import pytest

from packages.ai.inquiry import (
    CustomerInquiry,
    InquiryCategory,
    InquiryUrgency,
)
from packages.ai.schemas import (
    LLMResponse,
    TokenUsage,
)
from packages.ai.services.inquiry_service import (
    InquiryService,
)


class FakeLLMProvider:
    """
    Test implementation of our provider contract.
    """

    def __init__(
        self,
        category: InquiryCategory,
        requires_approval: bool = False,
    ) -> None:
        self.category = category
        self.requires_approval = requires_approval
        self.calls = 0

    def generate(
        self,
        prompt: str,
    ) -> LLMResponse:
        return LLMResponse(
            content="Fake AI response",
            model="fake-model",
            usage=TokenUsage(
                input_tokens=10,
                output_tokens=20,
                total_tokens=30,
                estimated_cost_usd=Decimal("0"),
            ),
        )

    def classify_inquiry(
        self,
        message: str,
    ) -> CustomerInquiry:
        self.calls += 1

        return CustomerInquiry(
            intent="customer_request",
            category=self.category,
            urgency=InquiryUrgency.MEDIUM,
            summary="Customer has a request.",
            suggested_action="Review the request.",
            requires_human_approval=self.requires_approval,
        )


def test_sales_inquiry_does_not_force_approval() -> None:
    provider = FakeLLMProvider(
        category=InquiryCategory.SALES,
    )

    service = InquiryService(provider)

    result = service.classify(
        "What is your subscription price?"
    )

    assert result.category == InquiryCategory.SALES
    assert result.requires_human_approval is False
    assert provider.calls == 1


def test_billing_requires_human_approval() -> None:
    provider = FakeLLMProvider(
        category=InquiryCategory.BILLING,
        requires_approval=False,
    )

    service = InquiryService(provider)

    result = service.classify(
        "I was charged twice."
    )

    assert result.requires_human_approval is True


def test_product_issue_requires_approval() -> None:
    provider = FakeLLMProvider(
        category=InquiryCategory.PRODUCT_ISSUE,
        requires_approval=False,
    )

    service = InquiryService(provider)

    result = service.classify(
        "My laptop is damaged."
    )

    assert result.requires_human_approval is True


def test_empty_message_does_not_call_provider() -> None:
    provider = FakeLLMProvider(
        category=InquiryCategory.GENERAL,
    )

    service = InquiryService(provider)

    with pytest.raises(ValueError):
        service.classify("   ")

    assert provider.calls == 0


def test_existing_approval_is_preserved() -> None:
    provider = FakeLLMProvider(
        category=InquiryCategory.SALES,
        requires_approval=True,
    )

    service = InquiryService(provider)

    result = service.classify(
        "Please approve a special sales discount."
    )

    assert result.requires_human_approval is True
