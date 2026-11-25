# vobench: Phase 1 model benchmark

This harness measures each candidate open-source model on the **target
computer** for:
- Malay pronunciation
- voice cloning
- emotional delivery
- reference-based style transfer (Mode A / Mode B)
- long-form stability
- VRAM, RAM and speed

Read `../docs/MODEL_RESEARCH.md` first for why these models were chosen.

## Steps
```bash
cd benchmark
bash envs/setup_envs.sh                 # one venv per model (their deps conflict)
cp refs/manifest.example.yaml refs/manifest.yaml   # then add your recordings (see refs/README.md)
venvs/voxcpm2/bin/vobench run --tts voxcpm2 --limit 3   # quick smoke test
bash run_all.sh                         # everything; resumable
```
Outputs go to `results/`:
- `<model>/…wav`: every generated file, plus `records.jsonl`. This records
  each job's settings, timing, peak VRAM/RAM and RTF, and every failure.
- `scores.jsonl`: automatic metrics.
- `REPORT.md`: tables per suite, plus **viability gates** from `gates.yaml`
  (edit the thresholds for your GPU).
- `listening.html`: a **blind** A/B/C rating page. It hides model names
  until you reveal them and exports a CSV. This is the deciding test.

## Suites (`suites/*.yaml`)
| Suite | What it answers |
|---|---|
| `malay` | 32 sentences: phonology, pepet/taling homographs, affixes, names, numbers, questions, code-switching; each run raw and normalised |
| `emotion` | all 17 tags in Malay, English controls, multi-tag transitions; each against a neutral baseline, for every emotion strategy the model supports |
| `cloning` | held-out text for each voice; VoxCPM2 in isolated and ultimate modes |
| `style_transfer` | Mode B routes `native`, `instruction`, `tts_then_vc`, `s2s` |
| `longform` | a 3–4 min Malay story with tag changes; chunk drift, chunk CER, memory |

## Metrics (eval venv; none of these ship in the app)
- **CER/WER**: faster-whisper large-v3 on Malay-normalised text.
- **Speaker similarity**: ECAPA cosine against a *held-out* real clip,
  compared with the human same-speaker ceiling.
- **UTMOS**: predicted naturalness.
- **emotion2vec**: SER label hit rate.
- **Delivery direction**: the change in loudness, pitch variability, speech
  rate and voicing relative to the neutral baseline.
- **Tag read aloud**: automatic fail.
- **Mode B**: identity leakage toward the reference speaker, style distance,
  and F0-contour correlation.

## Other commands
```bash
vobench sysinfo                                   # GPU / driver / RAM of this machine
vobench analyze ref.wav --transcript "..."        # structured speaking-style profile (JSON) + derived instruction
vobench anchors --suite suites/malay.yaml --dir refs/anchors/malay   # register human recordings
```

## Adding a model
Write `vobench/adapters/<name>.py` implementing `TTSAdapter.load/synthesize`,
set its capability flags, register it in `vobench/adapters/__init__.py`, and
add `envs/<name>.txt`. Planned next: MOSS-TTS v1.5. Its API needs checking on
a machine that can reach HuggingFace.

## Tests
`pip install -e ".[dev]" jiwer && pytest -q`. This covers tag parsing, the
Malay normaliser, chunking, style analysis on synthetic signals, consent
enforcement, and run→score→report plumbing with a test-only dummy adapter.
