# Architecture Notes: Voice AI Suite

## Pipeline

```text
Dashboard (pick a feature) ->
  ├─ Real-Time Agent      : Mic/LiveKit room -> STT -> LLM (+tools/memory) -> TTS -> audio out, interrupt-aware
  ├─ Agent Evaluation      : Persona script -> simulated caller -> agent under test -> recording+transcript -> LLM-judge scoring -> report
  ├─ Speech Model Evaluator: Reference dataset -> run across STT/TTS providers -> WER/latency/cost -> comparison report
  └─ Voice RAG Chatbot     : Mic/file input -> STT -> RAG retrieval -> LLM -> TTS -> playback
```

## Components

- `voice_core` — shared audio input abstraction (mic/file sources), Whisper
  STT wrapper, pluggable TTS backend wrapper, latency instrumentation,
  session/config handling
- `realtime_agent` feature app — ported from `realtime-voice-agent`
  (`src/rtva/`): LiveKit room/worker/token management (config, room, worker,
  token, livekit_client all ported as of this session; local audio capture
  goes through `voice_core.audio`'s shared `AudioSource` rather than a
  separate capture path), streaming STT/TTS, barge-in interruption handling
  and tool/function calling (`dialogue.py`'s bounded planner/tool-call loop,
  issue #7), and conversation memory (`memory.py`'s per-session, bounded
  `ConversationMemory`, issue #8) — `handle_user_turn` re-injects each
  session's prior turns into the planner as `history` on every call
- `agent_evaluation` feature app — ported from `voice-agent-evaluation`
  (`src/vae/`): persona scripting, simulated caller, LLM-as-judge scoring;
  can target `realtime_agent` as its agent-under-test once Phase 2 lands
- `speech_evaluator` feature app — ported from `speech-model-evaluator`
  (`src/sme/`): dataset/manifest assembly, provider adapters, WER/latency/cost
  metrics. Also has the Whisper size/version comparison harness
  (`comparison.py`, issue #12) and, as of 2026-09-19, the TTS provider/voice
  comparison harness (`tts_comparison.py`, issue #13) — synthesizes each
  reference text under every (provider, voice) via `voice_core.tts`,
  transcribes the result back via `voice_core.stt`, and scores intelligibility
  as WER against the original text, the same human-free proxy `comparison.py`
  uses for STT model sizes. Also has latency benchmarking and generation
  cost tracking (`benchmark.py`, issue #14) — times each STT/TTS comparison
  run with the Phase 1 `voice_core.latency.LatencyTracker` and pairs it with
  a documented per-second cost estimate (no real billing API in this
  environment), producing one `BenchmarkEntry` shape shared by both
  categories so they're directly comparable
- `voice_rag_chatbot` feature app — ported from `python-voice-rag-chatbot`
  (`src/voice_rag_chatbot/`): mic/file capture, transcription, RAG retrieval,
  LLM response, TTS playback. As of 2026-09-19, mic/file capture and RAG
  retrieval are in (issue #15), applying the knowledge-base "Chunking &
  embedding ingestion" pattern
  (`E:\Projects\LLM\knowledge-base\rag\chunking-and-embedding-ingestion.md`):
  `chunking.py`/`embeddings.py`/`vector_store.py`/`ingest.py` mirror that
  pattern's own reference implementation (`hybrid-research-agent/src/kb/`) —
  deterministic chunk ids from `hash(source, chunk_index, chunk_text)` for
  idempotent re-ingestion, a swappable `HashingEmbedder` (CPU-only, no paid
  API), and an atomically-written `JSONVectorStore`. `retrieval.py`'s
  `RagRetriever` feeds a `voice_core.audio.AudioSource` capture through
  `voice_core.stt.SpeechToText` and embeds the transcribed text to query the
  store — the same shared audio/STT layer every other feature app uses. As of
  2026-09-19, LLM response generation is in too (`generation.py`, issue #16):
  `ResponseGenerator` conditions a pluggable `LLMBackend` on the retrieved
  chunks, applying the new knowledge-base "Grounded generation with
  citation-or-fallback" pattern
  (`E:\Projects\LLM\knowledge-base\rag\grounded-generation-with-citations.md`,
  distilled from this issue since nothing matched closely beforehand) — a
  retrieval miss short-circuits to a fixed fallback without ever calling the
  LLM backend, and a grounded response carries structured `SourceCitation`s
  derived from chunk metadata rather than parsed out of the model's own text.

## Design Notes

- Registry pattern for feature apps (mirrors `medical-imaging-suite`'s
  `BaseImagingTask` / `@register_task` and `document-ai-suite`'s feature-app
  registry).
- All 4 source repos already wrap Whisper independently — `voice_core`
  centralizes that wrapper so only one STT integration needs
  maintaining/testing.
- Unlike `document-ai-suite`, there is no single "full pipeline" chain across
  all 4 features — `agent_evaluation` and `speech_evaluator` are *evaluation*
  tools for the other two, not sequential pipeline stages, so the dashboard
  is a plain 4-way feature picker (same shape as `video-analytics-suite` /
  `cv-suite`) with one cross-link: `agent_evaluation` can point its
  simulated-caller flow at the `realtime_agent` feature once both are ported.
- Reuse each source repo's existing Phase-1 modules directly where the
  interface fits `voice_core` rather than rewriting: `rtva.room`/`rtva.worker`/
  `rtva.token`, `vae.persona`/`vae.caller`, `sme.dataset`/`sme.librispeech`,
  `voice_rag_chatbot.audio_input`/`.transcription`/`.synthesis`.
