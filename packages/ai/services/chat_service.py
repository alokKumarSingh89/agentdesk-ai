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
from packages.ai.services.conversation_summarizer import (
    ConversationSummarizer,
)

class ChatService:

    def __init__(
        self,
        provider: LLMProvider,
        store: ConversationStore,
        max_history_messages: int = 10,
        max_context_tokens: int = 2000,
        token_counter: TokenCounter | None = None,
        summarizer: ConversationSummarizer | None = None,
        summarize_after_messages: int = 12,
        keep_recent_messages: int = 6,
    ) -> None:
        if max_history_messages < 1:
            raise ValueError(
                "History limit must be positive."
            )
        if max_context_tokens < 1:
            raise ValueError(
                "Context token budget must be positive."
            )
        if summarize_after_messages < 2:
            raise ValueError(
                "Summary threshold must be at least 2."
            )

        if keep_recent_messages < 1:
            raise ValueError(
                "Recent message count must be positive."
            )
        if (
            keep_recent_messages
            >= summarize_after_messages
        ):
            raise ValueError(
                "Recent message count must be lower "
                "than the summary threshold."
            )
        self.provider = provider
        self.store = store
        self.max_history_messages = max_history_messages
        self.token_counter = token_counter or TokenCounter()
        self.context_window = ContextWindow(
            token_counter=self.token_counter,
            max_tokens=max_context_tokens,
        )
        self.summarizer = (summarizer or ConversationSummarizer(provider))

        self.summarize_after_messages = summarize_after_messages

        self.keep_recent_messages = keep_recent_messages
    
    def create_conversation(self) -> UUID:
        conversation = self.store.create()

        return conversation.id
    def _summarize_if_needed(
        self,
        conversation_id: UUID,
    ) -> None:
        conversation = self.store.get(
            conversation_id
        )
        if (
            len(conversation.messages)
            < self.summarize_after_messages
        ):
            return
        split_index = (
            len(conversation.messages)
            - self.keep_recent_messages
        )
        messages_to_summarize = (
            conversation.messages[:split_index]
        )
        recent_messages = (
            conversation.messages[split_index:]
        )
        updated_summary = self.summarizer.summarize(
            messages=messages_to_summarize,
            existing_summary=conversation.summary,
        )
        
        self.store.update_summary(
            conversation_id,
            updated_summary,
        )
        
        self.store.replace_messages(
            conversation_id,
            recent_messages,
        )
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
        self._summarize_if_needed(
            conversation_id
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
        
        response = self.provider.chat(messages=messages, context=conversation.summary)

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