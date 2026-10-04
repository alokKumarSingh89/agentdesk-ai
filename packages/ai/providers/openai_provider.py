from openai import OpenAI
from packages.core.config import settings
from packages.ai.cost import calculate_llm_cost
from packages.ai.schemas import (
    LLMResponse,
    TokenUsage,
)

class OpenAIProvider:
    def __init__(self):
        self.client = OpenAI(
            api_key=settings.openai_api_key
        )
    
    def generate(self, prompt: str):
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