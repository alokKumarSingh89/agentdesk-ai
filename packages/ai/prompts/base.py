from dataclasses import dataclass

@dataclass(frozen=True)
class PromptTemplate:
    name: str
    version: str
    instructions: str
    
    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError(
                "Prompt name cannot be empty."
            )
        
        if not self.version.strip():
            raise ValueError(
                "Prompt version cannot be empty."
            )

        if not self.instructions.strip():
            raise ValueError(
                "Prompt instructions cannot be empty."
            )