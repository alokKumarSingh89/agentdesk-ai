import tiktoken

from packages.ai.memory.models import ChatMessage
from packages.core.config import settings

class TokenCounter:
    """
    Estimates token usage for conversation messages.

    This is a budgeting utility, not an exact
    replacement for provider usage metadata.
    """
    def __init__(self,model: str | None = None) -> None:
        self.model = model or settings.openai_model
        
        try:
            self.encoding = tiktoken.encoding_for_model(self.model)
        except KeyError:
            self.encoding = (
                tiktoken.get_encoding("o200k_base")
            )
            
    def count_text(self, text: str) -> int:
        return len(self.encoding.encode(text))
    
    def count_message(self,message: ChatMessage) -> int:
        # Approximate allowance for role and
        # message formatting overhead.
        overhead = 4
        
        return (
            self.count_text(message.role.value)
            + self.count_text(message.content)
            + overhead
        )
    def count_messages(self,messages: list[ChatMessage]) -> int:
        return sum(
            self.count_message(message)
            for message in messages
        )
    