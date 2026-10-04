"""OpenAI provider implementation."""
from openai import OpenAI
from packages.core.config import settings
from packages.ai.cost import calculate_llm_cost
from packages.ai.inquiry import CustomerInquiry
from packages.ai.prompts.registry import (
    prompt_registry,
)
from packages.ai.schemas import (
    LLMResponse,
    TokenUsage,
)

class OpenAIProvider:
    def __init__(self):
        self.client = OpenAI(
            api_key=settings.openai_api_key
        )
    
    def generate(self, prompt: str) -> LLMResponse:
        response = self.client.responses.create(
            model=settings.openai_model,
            input=prompt,
            instructions=(
                "You are AgentDesk AI, a business assistant. "
                "Introduce yourself as AgentDesk AI."
            ),
        )
        usage = response.usage
        if usage is None:
            raise RuntimeError(
                "LLM response is missing usage metadata."
            )
        
        estimated_cost = calculate_llm_cost(
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            input_price=settings.input_price_per_million,
            output_price=settings.output_price_per_million,
        )
        return LLMResponse(
            content=response.output_text,
            model=response.model,
            usage=TokenUsage(
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                total_tokens=usage.total_tokens,
                estimated_cost_usd=estimated_cost,
            )
        )
    def classify_inquiry(
        self,
        message: str,
    ) -> CustomerInquiry:
        """
        Classify a customer inquiry using
        structured LLM output.
        """
        if not message.strip():
            raise ValueError(
                "Customer message cannot be empty."
            )
        
        prompt = prompt_registry.get(
            name="customer_inquiry",
            version="1.0.0",
        )
        response = self.client.responses.parse(
            model=settings.openai_model,
            instructions=prompt.instructions,
            input=message,
            text_format=CustomerInquiry,
        )

        inquiry = response.output_parsed
        if inquiry is None:
            raise RuntimeError(
                "LLM did not return a valid inquiry. "
                "The response may have been refused "
                "or could not be parsed."
            )

        return inquiry