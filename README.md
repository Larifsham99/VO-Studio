# Latif Voice AI (VO-Studio)

A private, self-hosted expressive voice studio, built as a phased project.

**Current phase: 1, model research and benchmarking.** Per the development
strategy, the application (React + FastAPI + PyTorch) is built only after the
benchmark finds a viable model on the target computer.

- `docs/MODEL_RESEARCH.md`: candidate models, licences, hardware, how Mode A
  and Mode B map to real model capabilities, and decisions needed.
- `studio/`: **use it from an Android phone.** Your home PC runs the models and the phone controls it (guide in Malay: `studio/README.md`). This is a prototype until the benchmark confirms the best model.
- `benchmark/`: `vobench`, the harness that runs the benchmark on the target
  GPU and produces `REPORT.md` and a blind listening test.

Pieces of the eventual app already exist inside the harness, built and
tested to be reused later:
- emotion-tag parser and per-model control mapping
- Malay text normaliser
- long-form chunker
- reference style analyser
- model adapter interface
