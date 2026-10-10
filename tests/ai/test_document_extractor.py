
import pytest

from packages.ai.ingestion.extractor import (
    DocumentTextExtractor,
)


def test_extract_txt(tmp_path):
    file_path = tmp_path / "policy.txt"

    file_path.write_text(
        "Refunds take five days.",
        encoding="utf-8",
    )

    extractor = DocumentTextExtractor()

    assert extractor.extract(file_path) == (
        "Refunds take five days."
    )


def test_unsupported_extension(tmp_path):
    file_path = tmp_path / "policy.csv"

    file_path.write_text(
        "some data",
        encoding="utf-8",
    )

    extractor = DocumentTextExtractor()

    with pytest.raises(ValueError):
        extractor.extract(file_path)


def test_empty_document_rejected(tmp_path):
    file_path = tmp_path / "empty.txt"

    file_path.write_text(
        "   ",
        encoding="utf-8",
    )

    extractor = DocumentTextExtractor()

    with pytest.raises(ValueError):
        extractor.extract(file_path)
