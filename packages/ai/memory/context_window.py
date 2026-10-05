from packages.ai.memory.models import ChatMessage
from packages.ai.memory.token_counter import TokenCounter

class ContextWindow:
    def __init__(self,token_counter: TokenCounter,max_tokens: int = 2000,) -> None:
        if max_tokens <= 0:
            raise ValueError(
                "Token budget must be positive."
            )

        self.token_counter = token_counter
        self.max_tokens = max_tokens
    
    def select(self,messages: list[ChatMessage]) -> list[ChatMessage]:
        """
        Select the most recent messages that fit
        within the configured token budget.

        Preserve chronological order.

        Selection uses a contiguous recent suffix:
        older messages are not included if a newer
        message cannot fit.
        """
        
        selected: list[ChatMessage] = []

        used_tokens = 0
        for message in reversed(messages):
            message_tokens = (
                self.token_counter.count_message(message)
            )
            if (used_tokens + message_tokens> self.max_tokens):
                break
            selected.append(message)

            used_tokens += message_tokens
        selected.reverse()

        return selected