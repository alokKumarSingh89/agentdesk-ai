from packages.ai.prompts.base import PromptTemplate
from packages.ai.prompts.inquiry import (
    CUSTOMER_INQUIRY_V1,
)

class PromptRegistry:
    def __init__(self) -> None:
        self._prompts: dict[
            tuple[str, str],
            PromptTemplate,
        ] = {}
    def register(self,prompt: PromptTemplate) -> None:
        key = (prompt.name, prompt.version)
        if key in self._prompts:
            raise ValueError(
                f"Prompt already registered: {key}"
            )
        self._prompts[key] = prompt
    def get(self,name: str,version: str) -> PromptTemplate:
        key = (name, version)
        if key not in self._prompts:
            raise KeyError(
                f"Prompt not found: {name}@{version}"
            )

        return self._prompts[key]

prompt_registry = PromptRegistry()
prompt_registry.register(
    CUSTOMER_INQUIRY_V1
)