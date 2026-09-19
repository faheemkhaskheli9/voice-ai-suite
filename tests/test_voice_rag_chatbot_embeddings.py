import numpy as np
import pytest

from voice_rag_chatbot.embeddings import HashingEmbedder


def test_hashing_embedder_is_deterministic():
    embedder = HashingEmbedder(dimensions=64)
    v1 = embedder.embed(["the quick brown fox"])[0]
    v2 = embedder.embed(["the quick brown fox"])[0]
    assert np.array_equal(v1, v2)


def test_hashing_embedder_is_normalized():
    embedder = HashingEmbedder(dimensions=64)
    vec = embedder.embed(["some non-empty text here"])[0]
    assert abs(float(np.linalg.norm(vec)) - 1.0) < 1e-9


def test_different_text_gives_different_embedding():
    embedder = HashingEmbedder(dimensions=64)
    v1 = embedder.embed(["apples and oranges"])[0]
    v2 = embedder.embed(["quantum computing research"])[0]
    assert not np.array_equal(v1, v2)


def test_empty_text_is_zero_vector():
    embedder = HashingEmbedder(dimensions=64)
    vec = embedder.embed([""])[0]
    assert np.all(vec == 0)


def test_non_positive_dimensions_rejected():
    with pytest.raises(ValueError):
        HashingEmbedder(dimensions=0)
