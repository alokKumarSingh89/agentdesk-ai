import pytest

from packages.ai.prompts.base import PromptTemplate
from packages.ai.prompts.registry import (
    PromptRegistry,
    prompt_registry,
)

def test_registered_prompt_can_be_retrieved() -> None:
    prompt = prompt_registry.get(
        name="customer_inquiry",
        version="1.0.0",
    )

    assert prompt.name == "customer_inquiry"
    assert prompt.version == "1.0.0"
    assert len(prompt.instructions) > 0
    
def test_unknown_prompt_is_rejected() -> None:
    with pytest.raises(KeyError):
        prompt_registry.get(
            name="unknown_agent",
            version="1.0.0",
        )

def test_duplicate_registration_is_rejected() -> None:
    registry = PromptRegistry()

    prompt = PromptTemplate(
        name="test_agent",
        version="1.0.0",
        instructions="You are a test assistant.",
    )

    registry.register(prompt)

    with pytest.raises(ValueError):
        registry.register(prompt)

def test_empty_instructions_are_rejected() -> None:
    with pytest.raises(ValueError):
        PromptTemplate(
            name="invalid_prompt",
            version="1.0.0",
            instructions="",
        )