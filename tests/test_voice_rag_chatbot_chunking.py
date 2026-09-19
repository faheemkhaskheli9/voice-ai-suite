import pytest

from voice_rag_chatbot.chunking import chunk_text


def test_short_text_is_a_single_chunk():
    assert chunk_text("hello world", chunk_size=100, chunk_overlap=10) == ["hello world"]


def test_empty_text_yields_no_chunks():
    assert chunk_text("   ", chunk_size=100, chunk_overlap=10) == []


def test_long_text_is_split_with_overlap():
    text = " ".join(f"word{i}" for i in range(200))
    chunks = chunk_text(text, chunk_size=50, chunk_overlap=10)
    assert len(chunks) > 1
    assert all(len(c) <= 50 for c in chunks)
    assert "word0" in chunks[0]
    assert "word199" in chunks[-1]


def test_invalid_overlap_raises():
    with pytest.raises(ValueError):
        chunk_text("some text", chunk_size=10, chunk_overlap=10)
    with pytest.raises(ValueError):
        chunk_text("some text", chunk_size=0, chunk_overlap=0)


def test_chunking_terminates_on_pathological_overlap():
    text = "x" * 1000
    chunks = chunk_text(text, chunk_size=20, chunk_overlap=19)
    assert len(chunks) > 0
