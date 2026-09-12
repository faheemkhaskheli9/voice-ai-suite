# Voice AI Suite

> LLM, RAG & Agentic AI portfolio project — independent open-source implementation.
> This is an original, from-scratch build. It is not affiliated with, and does not
> contain any code, prompts, data, or business logic from, any employer or client.

![status](https://img.shields.io/badge/status-in%20progress-yellow)
![python](https://img.shields.io/badge/python-3.10%2B-blue)
![license](https://img.shields.io/badge/license-MIT-green)

## Combines

This is a flagship suite that will combine 4 voice/speech repos into one web
app with all 4 features selectable from a single dashboard UI — the same
combined-suite pattern already used in this portfolio for
`video-analytics-suite`, `medical-imaging-suite`, `trading-ai-suite`,
`cv-suite`, `automl-platform-suite`, `document-ai-suite`, and
`clinical-llm-suite`:

- [`realtime-voice-agent`](../realtime-voice-agent/) — LiveKit-based real-time
  two-way voice agent: STT -> LLM -> TTS with barge-in/interruption handling,
  tool calling, and cross-turn memory
- [`voice-agent-evaluation`](../voice-agent-evaluation/) — persona-driven
  simulated-caller batch testing of a voice agent, with LLM-as-judge scoring
  and pass/fail + latency reporting
- [`speech-model-evaluator`](../speech-model-evaluator/) — objective STT/TTS
  provider comparison: WER, latency, and cost benchmarking across models and
  vendors
- [`python-voice-rag-chatbot`](../python-voice-rag-chatbot/) — lightweight
  desktop mic -> RAG -> LLM -> TTS chatbot loop, without the real-time
  streaming complexity of the LiveKit agent

All 4 sit on the same STT/TTS/LLM voice-pipeline concept and already share
Whisper for transcription; `voice-agent-evaluation` and
`speech-model-evaluator` both exist specifically to *measure* the other two.
One dashboard, one shared audio/session layer, four selectable features
beats four near-duplicate STT/TTS plumbing implementations. The 4 originals
will get an archived banner + `status-archived` badge and move to
`E:\Projects\portfolio-archived-repos\` once this suite reaches feature
parity with each of them — no code or git history is deleted, only relocated.

## 1. Problem

Building a voice AI product touches the same four concerns every time: a
real-time conversational agent, a way to batch-test that agent with
simulated callers, a way to objectively compare the STT/TTS models it's
built on, and a lightweight reference implementation for demos that don't
need full real-time streaming. Splitting these into 4 repos means each one
re-implements its own audio capture, Whisper wrapper, and TTS wrapper. One
app, one shared voice-pipeline core, four picks from the dashboard.

## 2. Architecture

```text
Dashboard (pick a feature) ->
  ├─ Real-Time Agent      : Mic/LiveKit room -> STT -> LLM (+tools/memory) -> TTS -> audio out, interrupt-aware
  ├─ Agent Evaluation      : Persona script -> simulated caller -> agent under test -> recording+transcript -> LLM-judge scoring -> report
  ├─ Speech Model Evaluator: Reference dataset -> run across STT/TTS providers -> WER/latency/cost -> comparison report
  └─ Voice RAG Chatbot     : Mic/file input -> STT -> RAG retrieval -> LLM -> TTS -> playback
```

A shared `voice_core` layer (audio input abstraction for mic/file sources,
Whisper STT wrapper, pluggable TTS backend wrapper, latency instrumentation,
session/config handling) sits underneath 4 feature apps, each wrapping one
source repo's existing pipeline, registered via the same feature-registry
pattern used by this portfolio's other suites and selectable from one
dashboard.

## 3. Technology Stack

- Python, Django 5.x (feature-picker web app + run history)
- LiveKit (real-time audio pipeline, real-time agent feature)
- Whisper (STT, shared across all 4 features)
- TTS provider(s) — pyttsx3/gTTS for the offline-friendly path, pluggable
  cloud provider (e.g. ElevenLabs) for the real-time agent
- LangChain-style RAG retrieval (voice RAG chatbot feature)
- jiwer (WER), Matplotlib/Plotly (speech model evaluator reports)
- PostgreSQL in production, SQLite for local/dev (`DATABASE_URL` override)

## 4. Feature List

- **Real-Time Voice Agent** (from `realtime-voice-agent`): LiveKit real-time
  audio pipeline, streaming STT/TTS, interruption (barge-in) handling,
  conversation memory, tool/function calling mid-conversation, latency
  measurement
- **Voice Agent Evaluation** (from `voice-agent-evaluation`): configurable
  test personas, automated simulated conversations, recording + transcription,
  LLM-as-judge response scoring, per-turn latency, pass/fail flagging
- **Speech Model Evaluator** (from `speech-model-evaluator`): Whisper
  size/version comparison, TTS provider/voice comparison, WER measurement,
  latency benchmarking, generation cost tracking
- **Voice RAG Chatbot** (from `python-voice-rag-chatbot`): microphone/file
  input capture, speech recognition, RAG-based retrieval, LLM response
  generation, text-to-speech output
- Shared: one dashboard, one `voice_core` audio/STT/TTS layer, per-user run
  history across all 4 features

## 5. Implementation Plan

1. Phase 1: `voice_core` shared app (audio input abstraction, Whisper STT
   wrapper, pluggable TTS wrapper, latency instrumentation) + Django project
   skeleton with the dashboard shell
2. Phase 2: Port `realtime-voice-agent`'s LiveKit pipeline (`src/rtva/`) into
   a feature app — room/worker/token logic reused, interruption + tool
   calling + memory wired against `voice_core`
3. Phase 3: Port `voice-agent-evaluation`'s persona/caller pipeline
   (`src/vae/`) into a feature app that can target the real-time agent feature
   from Phase 2 as its agent-under-test
4. Phase 4: Port `speech-model-evaluator`'s dataset/provider-comparison
   pipeline (`src/sme/`) into a feature app, reusing its manifest format
5. Phase 5: Port `python-voice-rag-chatbot`'s mic/file -> RAG -> LLM -> TTS
   loop (`src/voice_rag_chatbot/`) into a feature app
6. Phase 6: Archive the 4 original repos (banner + badge, move to
   `portfolio-archived-repos`) once parity is confirmed

## Task Tracking

Work will be broken into phase-tagged user stories tracked as GitHub Issues,
not in this file. Implement Phase 1 issues first (later phases depend on it).
When you start one, add label `status:in-progress`. When you finish, close it
referencing the commit (e.g. `git commit -m "... Closes #4"`) and push.

## 6. Repository Structure

```text
voice-ai-suite/
├── README.md
├── LICENSE
├── .gitignore
├── pyproject.toml
├── .env.example
├── docker/
├── docs/
│   ├── architecture.md
│   └── evaluation.md
├── src/
├── tests/
├── configs/
├── scripts/
├── notebooks/
├── examples/
├── assets/
└── .github/
    └── workflows/
```

## 7. Setup

```bash
git clone <this-repo-url>
cd voice-ai-suite
python -m venv .venv && source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -r requirements.txt   # or: pip install -e .
cp .env.example .env              # fill in API keys / config
```

## 8. Dataset

Reference audio/text pairs for the Speech Model Evaluator feature come from
public corpora (e.g. LibriSpeech), same as `speech-model-evaluator`; not
vendored in this repo. No proprietary, employer-owned, or client-identifiable
data is used in this project.

## 9. Training / Execution

```bash
# Once Phase 1 lands:
python manage.py migrate
python manage.py runserver   # open http://127.0.0.1:8000/ and pick a feature
```

## 10. Evaluation

Document evaluation metrics and how to reproduce them here (see
`docs/evaluation.md`).

## 11. Results

_To be filled in as the implementation progresses — screenshots, metrics
tables, and sample outputs go here._

## 12. API

_If this project exposes an API, document the main endpoints here (or link to
auto-generated OpenAPI docs, e.g. `/docs` for FastAPI)._

## 13. Docker

```bash
docker build -t voice-ai-suite .
docker run -p 8000:8000 voice-ai-suite
```

## 14. Tests

```bash
pytest tests/
```

## 15. Limitations

- This is a from-scratch, independent recreation built for portfolio purposes.
- Performance numbers, once added, are based on public datasets and are not
  representative of any production system's real-world results.
- Scaffold stage: no code has been ported from the 4 source repos yet — see
  §5 Implementation Plan.

## 16. Future Work

- Port each source repo's existing Phase-1 code (all 4 have working CLIs
  already) rather than rewriting from scratch.
- Expand evaluation coverage and add CI-based regression checks.
- Track open items as GitHub Issues.

## 17. Disclosure

This repository is an **independent open-source recreation inspired by the
kind of production systems I have worked on professionally**. It contains no
employer or client source code, prompts, datasets, credentials, architecture
diagrams, or business logic. All code, data, and documentation here are
original or built on publicly available datasets and open-source tools.

---
_Last updated: 2026-09-12_
