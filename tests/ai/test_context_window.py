
import pytest

from packages.ai.memory.context_window import ContextWindow
from packages.ai.memory.models import (
    ChatMessage,
    MessageRole,
)


class FakeTokenCounter:
    """
    Deterministic counter for unit testing.

    Every character counts as one token.
    """

    def count_message(
        self,
        message: ChatMessage,
    ) -> int:
        return len(message.content)


def test_recent_messages_are_selected():
    counter = FakeTokenCounter()

    window = ContextWindow(
        token_counter=counter,
        max_tokens=10,
    )

    messages = [
        ChatMessage(
            role=MessageRole.USER,
            content="123456",
        ),
        ChatMessage(
            role=MessageRole.ASSISTANT,
            content="1234",
        ),
        ChatMessage(
            role=MessageRole.USER,
            content="12345",
        ),
    ]

    selected = window.select(messages)

    assert len(selected) == 2

    assert selected[0].content == "1234"
    assert selected[1].content == "12345"


def test_oversized_recent_message_is_excluded():
    counter = FakeTokenCounter()

    window = ContextWindow(
        token_counter=counter,
        max_tokens=5,
    )

    messages = [
        ChatMessage(
            role=MessageRole.USER,
            content="Old",
        ),
        ChatMessage(
            role=MessageRole.USER,
            content="This message is too long",
        ),
    ]

    selected = window.select(messages)

    assert selected == []


def test_invalid_token_budget():
    with pytest.raises(ValueError):
        ContextWindow(
            token_counter=FakeTokenCounter(),
            max_tokens=0,
        )
