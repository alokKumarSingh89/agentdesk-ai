from typing import Protocol, runtime_checkable

from packages.ai.inquiry import CustomerInquiry
from packages.ai.schemas import LLMResponse

@runtime_checkable
class LLMProvider(Protocol):
    """
    Contract implemented by supported LLM providers.
    """
    
    def generate(
        self,
        prompt: str,
    ) -> LLMResponse:
        ...

    def classify_inquiry(
        self,
        message: str,
    ) -> CustomerInquiry:
        ...