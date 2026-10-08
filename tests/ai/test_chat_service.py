
from decimal import Decimal

import pytest

from packages.ai.memory.models import (
    ChatMessage,
    MessageRole,
)
from packages.ai.memory.store import ConversationStore
from packages.ai.schemas import LLMResponse, TokenUsage
from packages.ai.services.chat_service import ChatService



class FakeChatProvider:

    def __init__(self) -> None:
        self.received_messages = []
        self.should_fail = False
        self.received_context = None

    def chat(
        self,
        messages: list[ChatMessage],
        context: str | None = None,
    ) -> LLMResponse:

        self.received_messages = [
            message.model_copy(deep=True)
            for message in messages
        ]
        self.received_context = context
        if self.should_fail:
            raise RuntimeError("Simulated API failure")

        return LLMResponse(
            content="This is a fake AI response.",
            model="fake-model",
            usage=TokenUsage(
                input_tokens=20,
                output_tokens=10,
                total_tokens=30,
                estimated_cost_usd=Decimal("0"),
            ),
        )


def test_conversation_remembers_previous_messages():
    provider = FakeChatProvider()
    store = ConversationStore()

    service = ChatService(
        provider=provider,
        store=store,
    )

    conversation_id = service.create_conversation()

    service.send_message(
        conversation_id,
        "What is RAG?",
    )

    service.send_message(
        conversation_id,
        "Give me an example of that.",
    )

    messages = provider.received_messages

    assert len(messages) == 3

    assert messages[0].content == "What is RAG?"

    assert messages[1].role == MessageRole.ASSISTANT

    assert (
        messages[2].content
        == "Give me an example of that."
    )


def test_failed_request_does_not_save_messages():
    provider = FakeChatProvider()
    store = ConversationStore()

    service = ChatService(
        provider=provider,
        store=store,
    )

    conversation_id = service.create_conversation()

    provider.should_fail = True

    with pytest.raises(RuntimeError):
        service.send_message(
            conversation_id,
            "Hello",
        )

    conversation = store.get(conversation_id)

    assert len(conversation.messages) == 0


def test_empty_message_is_rejected():
    provider = FakeChatProvider()
    store = ConversationStore()

    service = ChatService(
        provider=provider,
        store=store,
    )

    conversation_id = service.create_conversation()

    with pytest.raises(ValueError):
        service.send_message(
            conversation_id,
            "   ",
        )


def test_history_limit_is_applied():
    provider = FakeChatProvider()
    store = ConversationStore()

    service = ChatService(
        provider=provider,
        store=store,
        max_history_messages=2,
    )

    conversation_id = service.create_conversation()

    service.send_message(
        conversation_id,
        "First",
    )

    service.send_message(
        conversation_id,
        "Second",
    )

    service.send_message(
        conversation_id,
        "Third",
    )

    messages = provider.received_messages

    assert len(messages) == 3

    assert messages[0].role == MessageRole.USER
    assert messages[0].content == "Second"

    assert messages[1].role == MessageRole.ASSISTANT
    assert messages[1].content == (
        "This is a fake AI response."
    )

    assert messages[2].role == MessageRole.USER
    assert messages[2].content == "Third"
