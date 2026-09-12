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
  (`src/rtva/`): LiveKit room/worker/token management, streaming STT/TTS,
  interruption handling, tool calling, conversation memory
- `agent_evaluation` feature app — ported from `voice-agent-evaluation`
  (`src/vae/`): persona scripting, simulated caller, LLM-as-judge scoring;
  can target `realtime_agent` as its agent-under-test once Phase 2 lands
- `speech_evaluator` feature app — ported from `speech-model-evaluator`
  (`src/sme/`): dataset/manifest assembly, provider adapters, WER/latency/cost
  metrics
- `voice_rag_chatbot` feature app — ported from `python-voice-rag-chatbot`
  (`src/voice_rag_chatbot/`): mic/file capture, transcription, RAG retrieval,
  LLM response, TTS playback

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
