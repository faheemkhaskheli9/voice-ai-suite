import struct
import wave

import pytest

from voice_core.audio import FileAudioSource
from voice_core.stt import SpeechToText
from voice_core.tts import TextToSpeech
from voice_rag_chatbot.chatbot import VoiceRagChatbot
from voice_rag_chatbot.embeddings import HashingEmbedder
from voice_rag_chatbot.generation import NO_CONTEXT_RESPONSE, ResponseGenerator
from voice_rag_chatbot.ingest import IngestionPipeline
from voice_rag_chatbot.retrieval import RagRetriever
from voice_rag_chatbot.vector_store import JSONVectorStore

SAMPLE_RATE = 8000


class FakeWhisperBackend:
    def __init__(self, text: str):
        self.text = text

    def transcribe(self, samples, *, sample_rate, channels, model_size):
        return {"text": self.text}


class FakeLLMBackend:
    def __init__(self, response: str = "The lights are voice-controlled. [1]"):
        self.response = response

    def generate(self, prompt: str) -> str:
        return self.response


class FakeTTSBackend:
    def __init__(self, fail: bool = False):
        self.fail = fail
        self.last_text = None

    def synthesize(self, text: str, *, voice) -> bytes:
        self.last_text = text
        if self.fail:
            raise RuntimeError("provider crashed")
        return b"RIFF....WAVEfake-audio-bytes"


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
    IngestionPipeline(embedder=embedder, store=store, chunk_size=1000, chunk_overlap=0).ingest_directory(docs_dir)
    return store, embedder


def _chatbot(store, embedder, *, stt_text="turn on the lights", tts_fail=False, llm_response=None):
    stt = SpeechToText(FakeWhisperBackend(text=stt_text), model_size="base")
    retriever = RagRetriever(stt, embedder, store, top_k=2)
    kwargs = {} if llm_response is None else {"response": llm_response}
    generator = ResponseGenerator(FakeLLMBackend(**kwargs))
    tts = TextToSpeech(FakeTTSBackend(fail=tts_fail), provider="pyttsx3")
    return VoiceRagChatbot(retriever, generator, tts)


def test_full_loop_capture_transcribe_retrieve_generate_speak(tmp_path, populated_store):
    store, embedder = populated_store
    clip = tmp_path / "query.wav"
    _write_wav(clip)
    chatbot = _chatbot(store, embedder)

    result = chatbot.handle_audio_query(FileAudioSource(clip), tmp_path / "answer.wav")

    assert result.response.grounded is True
    assert "voice-controlled" in result.response.text
    assert result.audio_path is not None
    assert (tmp_path / "answer.wav").is_file()


def test_tts_output_uses_phase1_voice_core_tts_wrapper(tmp_path, populated_store):
    store, embedder = populated_store
    stt = SpeechToText(FakeWhisperBackend(text="turn on the lights"), model_size="base")
    retriever = RagRetriever(stt, embedder, store, top_k=2)
    generator = ResponseGenerator(FakeLLMBackend())
    tts_backend = FakeTTSBackend()
    tts = TextToSpeech(tts_backend, provider="pyttsx3", voice="default")
    chatbot = VoiceRagChatbot(retriever, generator, tts)

    result = chatbot.handle_text_query("turn on the lights", tmp_path / "answer.wav")

    assert tts_backend.last_text == result.response.text
    assert result.audio_path == str(tmp_path / "answer.wav")


def test_tts_failure_still_returns_text_response(tmp_path, populated_store):
    store, embedder = populated_store
    chatbot = _chatbot(store, embedder, tts_fail=True)

    result = chatbot.handle_text_query("turn on the lights", tmp_path / "answer.wav")

    assert result.audio_path is None
    assert result.response.grounded is True
    assert "voice-controlled" in result.response.text
    assert not (tmp_path / "answer.wav").exists()


def test_retrieval_miss_still_speaks_the_fallback_text(tmp_path):
    empty_store = JSONVectorStore(tmp_path / "empty.json")
    embedder = HashingEmbedder(dimensions=64)
    clip = tmp_path / "miss.wav"
    _write_wav(clip)
    chatbot = _chatbot(empty_store, embedder, stt_text="what is the capital of a distant planet")

    result = chatbot.handle_audio_query(FileAudioSource(clip), tmp_path / "answer.wav")

    assert result.response.grounded is False
    assert result.response.text == NO_CONTEXT_RESPONSE
    assert result.audio_path is not None
