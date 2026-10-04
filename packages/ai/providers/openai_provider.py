from openai import OpenAI
from packages.core.config import settings

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
        return response.output_text