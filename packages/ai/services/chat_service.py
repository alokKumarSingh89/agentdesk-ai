from uuid import UUID

from packages.ai.memory.models import (
    ChatMessage,
    MessageRole,
)
from packages.ai.memory.store import ConversationStore
from packages.ai.providers.base import LLMProvider
from packages.ai.schemas import LLMResponse

class ChatService:

    def __init__(
        self,
        provider: LLMProvider,
        store: ConversationStore,
        max_history_messages: int = 10,
    ) -> None:
        if max_history_messages < 1:
            raise ValueError(
                "History limit must be positive."
            )

        self.provider = provider
        self.store = store
        self.max_history_messages = max_history_messages
    
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

        # Include recent conversation context.
        # Always preserve the new user message.
        
        history = conversation.messages[
            -self.max_history_messages:
        ]
        
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