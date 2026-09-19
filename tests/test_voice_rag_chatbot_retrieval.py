import struct
import wave

import pytest

from voice_core.audio import FileAudioSource, MicAudioSource
from voice_core.stt import SpeechToText, TranscriptionError
from voice_rag_chatbot.embeddings import HashingEmbedder
from voice_rag_chatbot.ingest import IngestionPipeline
from voice_rag_chatbot.retrieval import RagRetriever
from voice_rag_chatbot.vector_store import JSONVectorStore

SAMPLE_RATE = 8000


class FakeWhisperBackend:
    """Test double standing in for a real Whisper model -- see
    voice_core/stt.py's module docstring."""

    def __init__(self, text: str, fail: bool = False):
        self.text = text
        self.fail = fail

    def transcribe(self, samples: bytes, *, sample_rate: int, channels: int, model_size: str) -> dict:
        if self.fail:
            raise RuntimeError("model crashed")
        return {"text": self.text}


class FakeMicBackend:
    def __init__(self, tone_amplitude: int = 1000):
        self.tone_amplitude = tone_amplitude

    def record(self, duration_seconds: float, sample_rate: int, channels: int) -> bytes:
        n_frames = int(duration_seconds * sample_rate)
        return struct.pack(f"<{n_frames * channels}h", *([self.tone_amplitude] * n_frames * channels))


def _write_wav(path, seconds=0.5, rate=SAMPLE_RATE):
    n = int(seconds * rate)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(struct.pack("<" + "h" * n, *([0] * n)))


@pytest.fixture
def populated_store(tmp_path):
    store = JSONVectorStore(tmp_path / "store.json")
    embedder = HashingEmbedder(dimensions=64)
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "lights.txt").write_text(
        "The smart lights can be turned on or off by voice command.", encoding="utf-8"
    )
    (docs_dir / "weather.txt").write_text(
        "The weather forecast API returns temperature and precipitation data.", encoding="utf-8"
    )
    IngestionPipeline(embedder=embedder, store=store, chunk_size=1000, chunk_overlap=0).ingest_directory(docs_dir)
    return store, embedder


def test_retrieve_from_file_audio_reuses_voice_core_audio_abstraction(tmp_path, populated_store):
    store, embedder = populated_store
    clip = tmp_path / "query.wav"
    _write_wav(clip)
    stt = SpeechToText(FakeWhisperBackend(text="turn on the lights"), model_size="base")
    retriever = RagRetriever(stt, embedder, store, top_k=2)

    result = retriever.retrieve_from_audio(FileAudioSource(clip))

    assert result.query_text == "turn on the lights"
    assert result.chunks
    assert "lights" in result.chunks[0].document.lower()


def test_retrieve_from_mic_audio(tmp_path, populated_store):
    store, embedder = populated_store
    stt = SpeechToText(FakeWhisperBackend(text="what is the weather forecast"), model_size="base")
    retriever = RagRetriever(stt, embedder, store, top_k=2)

    result = retriever.retrieve_from_audio(MicAudioSource(FakeMicBackend(), duration_seconds=0.5))

    assert result.query_text == "what is the weather forecast"
    assert "weather" in result.chunks[0].document.lower()


def test_retrieve_from_text_skips_transcription(populated_store):
    store, embedder = populated_store
    stt = SpeechToText(FakeWhisperBackend(text="unused"), model_size="base")
    retriever = RagRetriever(stt, embedder, store, top_k=1)

    result = retriever.retrieve_from_text("smart lights voice command")

    assert result.query_text == "smart lights voice command"
    assert len(result.chunks) == 1


def test_empty_query_returns_no_chunks(populated_store):
    store, embedder = populated_store
    stt = SpeechToText(FakeWhisperBackend(text="unused"), model_size="base")
    retriever = RagRetriever(stt, embedder, store)

    assert retriever.retrieve_from_text("   ").chunks == ()


def test_empty_store_returns_no_chunks(tmp_path):
    store = JSONVectorStore(tmp_path / "empty.json")
    embedder = HashingEmbedder(dimensions=64)
    stt = SpeechToText(FakeWhisperBackend(text="unused"), model_size="base")
    retriever = RagRetriever(stt, embedder, store)

    assert retriever.retrieve_from_text("anything").chunks == ()


def test_transcription_failure_propagates_instead_of_being_treated_as_empty(tmp_path, populated_store):
    store, embedder = populated_store
    clip = tmp_path / "query.wav"
    _write_wav(clip)
    stt = SpeechToText(FakeWhisperBackend(text="unused", fail=True), model_size="base")
    retriever = RagRetriever(stt, embedder, store)

    with pytest.raises(TranscriptionError):
        retriever.retrieve_from_audio(FileAudioSource(clip))


def test_score_threshold_filters_weak_matches(populated_store):
    store, embedder = populated_store
    stt = SpeechToText(FakeWhisperBackend(text="unused"), model_size="base")
    retriever = RagRetriever(stt, embedder, store, top_k=5, score_threshold=1.0)

    assert retriever.retrieve_from_text("turn on the lights").chunks == ()


def test_invalid_top_k_rejected(populated_store):
    store, embedder = populated_store
    stt = SpeechToText(FakeWhisperBackend(text="unused"), model_size="base")
    with pytest.raises(ValueError):
        RagRetriever(stt, embedder, store, top_k=0)
