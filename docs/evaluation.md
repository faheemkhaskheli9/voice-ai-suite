# Evaluation Notes: Voice AI Suite

## Metrics

- **Real-Time Agent**: end-to-end turn latency (STT start -> TTS first audio
  byte), interruption/barge-in success rate, tool-call success rate
- **Agent Evaluation**: pass/fail rate per persona, LLM-judge score
  distribution, per-turn latency, failed-test flag rate
- **Speech Model Evaluator**: word error rate (WER) per STT model/version,
  TTS provider/voice comparison (subjective + latency), generation cost per
  minute of audio
- **Voice RAG Chatbot**: retrieval relevance (RAG hit rate), end-to-end
  response latency, transcription accuracy on captured audio

## Reproducing Results

```bash
python -m src.evaluate --config configs/eval.yaml
```

## Result Log

| Date | Config | Metric | Value | Notes |
|------|--------|--------|-------|-------|
|      |        |        |       |       |
