
import pytest

from packages.ai.ingestion.chunker import TextChunker


def test_short_text_creates_one_chunk():
    chunker = TextChunker(
        chunk_size=100,
        chunk_overlap=10,
    )

    chunks = chunker.split(
        "Refunds take five business days."
    )

    assert len(chunks) == 1


def test_long_text_creates_multiple_chunks():
    chunker = TextChunker(
        chunk_size=20,
        chunk_overlap=5,
    )

    text = "Refund policy. " * 30

    chunks = chunker.split(text)

    assert len(chunks) > 1


def test_empty_text_returns_no_chunks():
    chunker = TextChunker()

    assert chunker.split("   ") == []


def test_invalid_overlap_rejected():
    with pytest.raises(ValueError):
        TextChunker(
            chunk_size=10,
            chunk_overlap=10,
        )
