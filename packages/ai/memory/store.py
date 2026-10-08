from uuid import UUID

from packages.ai.memory.models import (
    ChatMessage,
    Conversation,
)

class ConversationStore:
    """
    Temporary in-memory conversation storage.

    Later, we can replace this implementation
    with PostgreSQL or Redis.
    """
    
    def __init__(self) -> None:
        self._conversations: dict[UUID,Conversation,] = {}
    
    def create(self) -> Conversation:
        conversation = Conversation()
        self._conversations[conversation.id] = conversation
        
        return conversation.model_copy(deep=True)
    
    def get(
        self,
        conversation_id: UUID,
    ) -> Conversation:
        conversation = self._conversations.get(
            conversation_id
        )
        
        if conversation is None:
            raise KeyError(
                "Conversation not found."
            )

        return conversation.model_copy(deep=True)
    
    def add_message(
        self,
        conversation_id: UUID,
        message: ChatMessage,
    ) -> None:
        conversation = self._conversations.get(
            conversation_id
        )
        
        if conversation is None:
            raise KeyError(
                "Conversation not found."
            )

        conversation.messages.append(
            message.model_copy(deep=True)
        )
        
    def replace_messages(
        self,
        conversation_id: UUID,
        messages: list[ChatMessage],
    ) -> None:
        conversation = self._conversations.get(
            conversation_id
        )
        if conversation is None:
            raise KeyError(
                "Conversation not found."
            )

        conversation.messages = [
            message.model_copy(deep=True)
            for message in messages
        ]
    
    def update_summary(
        self,
        conversation_id: UUID,
        summary: str,
    ) -> None:
        conversation = self._conversations.get(
            conversation_id
        )
        if conversation is None:
            raise KeyError(
                "Conversation not found."
            )

        summary = summary.strip()
        if not summary:
            raise ValueError(
                "Conversation summary cannot be empty."
            )

        conversation.summary = summary