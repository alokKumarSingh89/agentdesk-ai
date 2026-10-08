from decimal import Decimal
import pytest
from packages.ai.memory.models import (
    ChatMessage,
)
from packages.ai.memory.store import ConversationStore
from packages.ai.schemas import (
    LLMResponse,
    TokenUsage,
)
from packages.ai.services.chat_service import ChatService


class FakeTokenCounter:

    def count_message(
        self,
        message: ChatMessage,
    ) -> int:
        return 1
    
class FakeProvider:

    def __init__(self) -> None:
        self.received_context = None
        
    def generate(
        self,
        prompt: str,
    ) -> LLMResponse:
        return LLMResponse(
            content=(
                "Customer previously reported "
                "a billing problem."
            ),
            model="fake-model",
            usage=TokenUsage(
                input_tokens=10,
                output_tokens=10,
                total_tokens=20,
                estimated_cost_usd=Decimal("0"),
            ),
        )
    def chat(
            self,
            messages: list[ChatMessage],
            context: str | None = None,
        ) -> LLMResponse:
            self.received_context = context
            return LLMResponse(
                content="Fake response",
                model="fake-model",
                usage=TokenUsage(
                    input_tokens=10,
                    output_tokens=10,
                    total_tokens=20,
                    estimated_cost_usd=Decimal("0"),
                ),
            )
            
def test_old_messages_are_summarized():
    provider = FakeProvider()
    store = ConversationStore()
    
    service = ChatService(
        provider=provider,
        store=store,
        max_context_tokens=100,
        token_counter=FakeTokenCounter(),
        summarize_after_messages=4,
        keep_recent_messages=2,
    )

    conversation_id = (
        service.create_conversation()
    )

    service.send_message(
        conversation_id,
        "I have a billing problem.",
    )

    service.send_message(
        conversation_id,
        "I was charged twice.",
    )

    # Four messages now exist:
    #
    # user
    # assistant
    # user
    # assistant
    #
    # The next request triggers summarization.

    service.send_message(
        conversation_id,
        "What should I do?",
    )

    conversation = store.get(
        conversation_id
    )

    assert conversation.summary == (
        "Customer previously reported "
        "a billing problem."
    )

    # Two recent messages were retained,
    # then the new user/assistant pair was saved.
    assert len(conversation.messages) == 4

def test_summary_is_sent_as_context():
    provider = FakeProvider()
    store = ConversationStore()

    service = ChatService(
        provider=provider,
        store=store,
        max_context_tokens=100,
        token_counter=FakeTokenCounter(),
        summarize_after_messages=4,
        keep_recent_messages=2,
    )

    conversation_id = (
        service.create_conversation()
    )

    service.send_message(
        conversation_id,
        "Billing issue.",
    )

    service.send_message(
        conversation_id,
        "Charged twice.",
    )

    service.send_message(
        conversation_id,
        "Do you remember my issue?",
    )

    assert provider.received_context == (
        "Customer previously reported "
        "a billing problem."
    )

class FailingSummaryProvider(FakeProvider):

    def generate(
        self,
        prompt: str,
    ) -> LLMResponse:
        raise RuntimeError(
            "Summary generation failed"
        )


def test_failed_summary_keeps_original_messages():
    provider = FailingSummaryProvider()

    store = ConversationStore()

    service = ChatService(
        provider=provider,
        store=store,
        max_context_tokens=100,
        token_counter=FakeTokenCounter(),
        summarize_after_messages=4,
        keep_recent_messages=2,
    )

    conversation_id = (
        service.create_conversation()
    )

    service.send_message(
        conversation_id,
        "First",
    )

    service.send_message(
        conversation_id,
        "Second",
    )

    before = store.get(
        conversation_id
    )

    assert len(before.messages) == 4

    with pytest.raises(RuntimeError):
        service.send_message(
            conversation_id,
            "Third",
        )

    after = store.get(
        conversation_id
    )

    assert after.summary is None

    assert len(after.messages) == 4