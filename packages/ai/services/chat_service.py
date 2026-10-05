from uuid import UUID

from packages.ai.memory.models import (
    ChatMessage,
    MessageRole,
)
from packages.ai.memory.store import ConversationStore
from packages.ai.providers.base import LLMProvider
from packages.ai.schemas import LLMResponse
from packages.ai.memory.token_counter import TokenCounter
from packages.ai.memory.context_window import ContextWindow

class ChatService:

    def __init__(
        self,
        provider: LLMProvider,
        store: ConversationStore,
        max_history_messages: int = 10,
        max_context_tokens: int = 2000,
        token_counter: TokenCounter | None = None,
    ) -> None:
        if max_history_messages < 1:
            raise ValueError(
                "History limit must be positive."
            )
        if max_context_tokens < 1:
            raise ValueError(
                "Context token budget must be positive."
            )

        self.provider = provider
        self.store = store
        self.max_history_messages = max_history_messages
        self.token_counter = (
            token_counter or TokenCounter()
        )
        self.context_window = ContextWindow(
            token_counter=self.token_counter,
            max_tokens=max_context_tokens,
        )
    
    def create_conversation(self) -> UUID:
        conversation = self.store.create()

        return conversation.id
    
    def send_message(
        self,
        conversation_id: UUID,
        content: str,
    ) -> LLMResponse:

        content = content.strip()

        if not content:
            raise ValueError(
                "Message cannot be empty."
            )

        conversation = self.store.get(
            conversation_id
        )
        user_message = ChatMessage(
            role=MessageRole.USER,
            content=content,
        )

        # The current user message has priority.
        current_tokens = (
            self.token_counter.count_message(
                user_message
            )
        )
        if (
            current_tokens
            > self.context_window.max_tokens
        ):
            raise ValueError(
                "Message exceeds the configured "
                "conversation token budget."
            )
        
        # Reserve tokens for the current message.
        history_budget = (
            self.context_window.max_tokens
            - current_tokens
        )
        
        history_window = ContextWindow(
            token_counter=self.token_counter,
            max_tokens=max(1, history_budget),
        )
        
        recent_history = conversation.messages[
            -self.max_history_messages:
        ]
        
        history = (
            history_window.select(recent_history)
            if history_budget > 0
            else []
        )
        messages = [
            *history,
            user_message,
        ]
        
        response = self.provider.chat(messages)

        # Persist only after a successful LLM call.
        self.store.add_message(
            conversation_id,
            user_message,
        )
        self.store.add_message(
            conversation_id,
            ChatMessage(
                role=MessageRole.ASSISTANT,
                content=response.content,
            ),
        )
        return response