import pytest

from voice_rag_chatbot.generation import (
    NO_CONTEXT_RESPONSE,
    GenerationError,
    ResponseGenerator,
)
from voice_rag_chatbot.retrieval import RetrievalResult, RetrievedChunk


class FakeLLMBackend:
    def __init__(self, response: str = "The lights can be voice-controlled. [1]", fail: bool = False):
        self.response = response
        self.fail = fail
        self.last_prompt = None

    def generate(self, prompt: str) -> str:
        self.last_prompt = prompt
        if self.fail:
            raise RuntimeError("provider crashed")
        return self.response


def _grounded_result():
    return RetrievalResult(
        query_text="can the lights be controlled by voice",
        chunks=(
            RetrievedChunk(
                document="The smart lights can be turned on or off by voice command.",
                score=0.92,
                metadata={"source": "docs/lights.txt", "title": "lights", "chunk_index": 0},
            ),
        ),
    )


def test_generates_response_conditioned_on_retrieved_context():
    backend = FakeLLMBackend()
    generator = ResponseGenerator(backend)

    response = generator.generate(_grounded_result())

    assert response.grounded is True
    assert "voice-controlled" in response.text
    assert "lights.txt" in backend.last_prompt
    assert "can the lights be controlled by voice" in backend.last_prompt


def test_response_references_its_source_chunks():
    generator = ResponseGenerator(FakeLLMBackend())

    response = generator.generate(_grounded_result())

    assert len(response.sources) == 1
    assert response.sources[0].source == "docs/lights.txt"
    assert "Sources: docs/lights.txt" in response.text


def test_retrieval_miss_falls_back_to_generic_response_without_calling_backend():
    backend = FakeLLMBackend()
    generator = ResponseGenerator(backend)
    empty_result = RetrievalResult(query_text="something totally unrelated", chunks=())

    response = generator.generate(empty_result)

    assert response.text == NO_CONTEXT_RESPONSE
    assert response.grounded is False
    assert response.sources == ()
    assert backend.last_prompt is None  # never called -- no chance of a fabricated citation


def test_backend_failure_propagates_instead_of_being_silently_ungrounded():
    generator = ResponseGenerator(FakeLLMBackend(fail=True))

    with pytest.raises(GenerationError):
        generator.generate(_grounded_result())


def test_backend_returning_blank_text_is_rejected():
    generator = ResponseGenerator(FakeLLMBackend(response="   "))

    with pytest.raises(GenerationError):
        generator.generate(_grounded_result())


def test_multiple_source_chunks_are_all_cited():
    backend = FakeLLMBackend(response="Lights are voice controlled and weather is available. [1][2]")
    generator = ResponseGenerator(backend)
    result = RetrievalResult(
        query_text="lights and weather",
        chunks=(
            RetrievedChunk(document="lights info", score=0.9, metadata={"source": "docs/lights.txt"}),
            RetrievedChunk(document="weather info", score=0.8, metadata={"source": "docs/weather.txt"}),
        ),
    )

    response = generator.generate(result)

    assert len(response.sources) == 2
    assert "docs/lights.txt" in response.text
    assert "docs/weather.txt" in response.text
