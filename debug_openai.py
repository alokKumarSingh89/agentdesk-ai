
from openai import OpenAI

from packages.core.config import settings

client = OpenAI(api_key=settings.openai_api_key)

response = client.responses.create(
    model=settings.openai_model,
    instructions=(
        "You are AgentDesk AI, a business assistant. "
        "Introduce yourself as AgentDesk AI."
    ),
    input="Hello, introduce yourself in one sentence.",
)

print(response.output_text)
