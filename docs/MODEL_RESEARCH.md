# Latif Voice AI: Phase 1 model research

Status: **research done, benchmark not yet run.** Researched on 2026-11-25.
Nothing in this document is a measured result on the target computer. Anything
marked *vendor* comes from the model's own documentation. Anything marked
*estimate* is my own reasoning. The benchmark in `benchmark/` produces the real
numbers. Per the development strategy, the full application is not built until
those numbers show that a viable model exists.

## 1. Bottom line (hypothesis to be confirmed by the benchmark)

| Role | First choice | Backup / comparison |
|---|---|---|
| Main TTS engine (Malay, cloning, emotion) | **VoxCPM2** (Apache-2.0) | Chatterbox Multilingual (MIT); MOSS-TTS v1.5 (Apache-2.0) |
| Malay pronunciation specialist | Mesolitica Malaysian-F5-TTS-v3 (**non-commercial**) | VoxCPM2 + LoRA on Malay data |
| Mode A (clone voice and style from a reference) | VoxCPM2 "ultimate cloning" (reference audio + transcript) | Chatterbox, F5-Malaysian |
| Mode B (keep a permanent voice, take delivery from a reference) | Route test: VoxCPM2 style-continuation vs. TTS→Seed-VC v2 vs. speech-to-speech VC | Chatterbox VC (MIT) |
| Permanent voice profiles | LoRA fine-tune of the main engine on each voice (vendor: 5–10 min of audio) | Zero-shot reference only |

What matters most: **no open model available today does ElevenLabs-style
inline emotion tags in Malay out of the box.** The closest one, Fish Audio S2
Pro, has free-form `[tags]` and covers 80+ languages, but its licence is
research and non-commercial only. My recommended emotion approach uses three
real control mechanisms, and the benchmark compares them directly:

1. **Natural-language instruction.** VoxCPM2's "controllable cloning" keeps
   the cloned timbre and follows a style instruction such as
   `(sad, low and heavy, slower pace)`.
2. **Per-voice emotion bank.** Each permanent speaker records short clips
   performing each emotion. A zero-shot cloning model copies the delivery of
   its reference clip, so `[sad]` means "use voice A's *sad* clip as the
   reference". This is real performance by the model, not post-processing.
3. **LoRA fine-tune on emotional recordings** of each permanent voice, with
   the emotion conditioned by instruction. This is the most likely route to
   ElevenLabs-level conviction in Malay, but it needs about 30–60 min of
   labelled emotional speech per voice (*estimate*).

Tags are always parsed out of the text and turned into control signals
(`benchmark/vobench/text/emotion_tags.py`). The benchmark marks any output
whose ASR transcript contains a tag word as an automatic failure.

## 2. Candidate matrix

| Model | Malay | Zero-shot clone | Emotion control | Separate style ref (Mode B) | Long-form | Licence (weights) | VRAM |
|---|---|---|---|---|---|---|---|
| **VoxCPM2** (OpenBMB, 2B) | ✅ one of 30 languages | ✅ isolated ref, or continuation | ✅ instruction in `(…)`, voice design | ⚠️ maybe: isolated timbre ref + continuation prompt *(untested)* | streaming | **Apache-2.0** | ~8 GB *(vendor)*; RTF ~0.30 on 4090 |
| **Chatterbox Multilingual** (Resemble, 0.5B) | ✅ `ms`, one of 23 | ✅ | ⚠️ exaggeration scalar only | ❌ (separate Chatterbox VC exists) | manual chunking | **MIT** (adds Perth watermark) | ~4–6 GB *(estimate)* |
| **MOSS-TTS v1.5** (OpenMOSS, 8B) | ✅ one of 31 | ✅ | ✅ instruction; pause markers | unknown | ✅ designed for long-form/dialogue | **Apache-2.0** (from mirrors; check the original card) | ~16–20 GB bf16 *(estimate)*; vendor says the llama.cpp build fits 8 GB |
| **Malaysian-F5-TTS-v3** (Mesolitica) | ✅✅ Malay, Malaysian English, Mandarin, code-switching | ✅ needs transcript | ⚠️ only via the reference clip | ❌ | chunking | **CC-BY-NC-4.0** ❌ commercial | ~3–6 GB *(estimate)* |
| Mesolitica Malaysian-TTS 0.6B/1.7B/4B | ✅✅ | partial | unknown | ❌ | ? | not verified | – |
| **Fish Audio S2 Pro** | ✅ (80+ languages, tiers unclear) | ✅ | ✅✅ free-form `[tags]` | ❌ | ✅ | **Fish Audio Research Licence**, non-commercial ❌ | large *(not verified)* |
| **IndexTTS2** (bilibili) | ❌ zh/en | ✅ | ✅✅ separate emotion reference, emotion vectors, text | ✅✅ timbre and emotion fully disentangled | ✅ | bilibili Model Use Licence (custom, revocable) | ~8–12 GB *(estimate)* |
| **Qwen3-TTS** (Alibaba) | ❌ 10 languages, no Malay | ✅ | ✅ instruct (CustomVoice/VoiceDesign) | ❌ | ✅ | Apache-2.0 | 1.7B: ~6–8 GB *(estimate)* |
| XTTS-v2 (Coqui) | ❌ | ✅ | ❌ | ❌ | chunking | CPML non-commercial; vendor defunct | ~4 GB |
| Dia / Orpheus / Chatterbox-Turbo / Maya1 | ❌ English | varies | ✅ nonverbal tags | ❌ | – | Apache/MIT | – |
| **Seed-VC v2** (voice conversion) | language-agnostic | ✅ target timbre | keeps source prosody (`convert_style=False`) | ✅ Mode B via VC | chunked | **GPL-3.0** | ~4–6 GB *(estimate)* |
| **Chatterbox VC** | language-agnostic | ✅ | keeps source prosody | ✅ Mode B via VC | – | MIT | ~3 GB *(estimate)* |

Python APIs were checked against the actual PyPI packages: chatterbox-tts
0.1.7, voxcpm 2.0.3, f5-tts 1.1.22, seed-vc 0.4.3, qwen-tts 0.1.1. MOSS-TTS and
IndexTTS2 have no PyPI package, and HuggingFace was unreachable from this
environment. Their adapters are therefore **not written yet** (see §8) rather
than guessed.

### Why not use the most expressive models as the engine
- **IndexTTS2** has the best design for Mode B: separate timbre and emotion
  references, and an emotion strength control. But it does not speak Malay,
  and its custom licence is revocable. It is the architecture reference, and
  a Malay fine-tune would be a later research project.
- **Fish Audio S2 Pro** comes closest to the ElevenLabs tag experience. Its
  licence rules it out for anything commercial. If the studio will only ever
  be personal and non-commercial, it is worth adding as a quality-ceiling
  comparison (decision needed, see §9).

## 3. Licences and what they mean for this project

| Component | Licence | Private self-hosted use | Commercial use / distribution |
|---|---|---|---|
| VoxCPM2 | Apache-2.0 | ✅ | ✅ (card forbids impersonation and fraud) |
| Chatterbox ML / VC | MIT | ✅ | ✅. Outputs carry an inaudible Perth watermark |
| MOSS-TTS v1.5 | Apache-2.0 *(verify the original card)* | ✅ | ✅ if confirmed |
| Qwen3-TTS | Apache-2.0 | ✅ | ✅ |
| Malaysian-F5-TTS (all versions), F5-TTS base | CC-BY-NC-4.0 | ✅ non-commercial only | ❌ |
| Fish Audio S2 Pro | Fish Audio Research Licence | ✅ research / non-commercial | ❌ without a paid licence |
| IndexTTS2 | bilibili Model Use Licence | ✅ under its terms | ⚠️ revenue threshold, revocable |
| XTTS-v2 | Coqui Public Model Licence | ✅ non-commercial | ❌ |
| Seed-VC | GPL-3.0 | ✅ | ✅ but a distributed app that links it must be GPL |
| Eval-only tools (Whisper, SpeechBrain ECAPA, UTMOS, emotion2vec) | MIT / Apache / various | benchmark only, not shipped | – |

None of these has a per-request fee. All of them run offline once the weights
are downloaded.

## 4. Hardware requirements

These are vendor figures and estimates. The benchmark records the real peak
VRAM (NVML, the whole process), peak torch-allocated memory, peak RAM, load
time and RTF for every generation.

| Configuration | GPU VRAM | System RAM | Notes |
|---|---|---|---|
| Minimum (one 2B engine, fp16/bf16) | 8 GB | 16 GB | VoxCPM2 vendor figure ~8 GB is tight; no VC at the same time |
| **Recommended** | 12–16 GB (e.g. RTX 4070 Ti / 4080 / 5070 Ti) | 32 GB | engine + VC resident; LoRA training of 2B at small batch |
| Comfortable / MOSS-8B in bf16 | 24 GB (RTX 3090 / 4090 / 5090) | 64 GB | all candidates loaded together; full fine-tunes |
| CPU only | – | 32 GB | expect RTF well above 1 (*estimate*); not viable for an "ElevenLabs-like" experience, except MOSS-TTS-Nano (vendor: streams on 4 CPU cores) |

Apple Silicon (MPS) works for Chatterbox. Other models are not verified on
MPS. Disk: about 5–20 GB per model, plus about 10 GB for the eval models.

## 5. How Mode A and Mode B map to real model capabilities

**Mode A (authorised reference: identity and style)**
- Reference upload, then analysis (`vobench analyze`: pitch, rhythm, rate,
  pauses, emphasis, emotion timeline), then Whisper transcript, then VoxCPM2
  *ultimate cloning* (reference used as timbre reference and as continuation
  prompt). This copies timbre, rhythm and emotion as the vendor describes.
- The UI must store a consent record per reference. The harness already
  refuses any recording without one.

**Mode B (permanent identity, delivery from a reference)**, four routes
benchmarked:

| Route | How | Works for new text? | Expected risk |
|---|---|---|---|
| `native` | VoxCPM2: `reference_wav`=permanent voice, continuation prompt=style reference | ✅ | identity leaking from the continuation prompt |
| `instruction` | analyse the reference → style instruction → controllable clone of the permanent voice | ✅ | coarse transfer; loses fine timing |
| `tts_then_vc` | clone the *reference speaker* on new text, then Seed-VC v2 / Chatterbox VC to the permanent voice | ✅ | two-model artifacts; uses the reference speaker's identity temporarily, so their consent is still needed |
| `s2s` | voice-convert the reference recording itself | ❌ same words only (re-voicing / dubbing) | best prosody fidelity, limited use |

Mode B scoring uses three numbers together:
- similarity to the **permanent** voice (should be high);
- similarity to the **reference speaker** (should be low, which shows the
  identity did not leak);
- text-independent **style distance** to the reference. For same-text cases,
  DTW-aligned F0-contour correlation is added.

**Pitch-shifting, speed changes and pre-recorded inserts are never used.**
The reference analysis produces a description and conditioning signals; the
model itself produces the performance. Long-form chunks are joined with
short silences only.

## 6. Malay-specific findings and risks
- **Text front-end is required.** Raw digits, `RM25.50`, dates, `ke-3` and
  abbreviations are where multilingual models usually fail. I wrote a Malay
  normaliser (`vobench/text/malay_normalize.py`, unit-tested). The benchmark
  runs every Malay item both raw and normalised to measure how much it helps.
- **Pepet and taling homographs** (*perang* war/blond, *semak* check/bushes)
  cannot be resolved from spelling. The suite tests them in context. A
  pronunciation lexicon or hint mechanism may be needed later.
- Code-switching (Malaysian English mixed with Malay) and particles
  (*lah, ke, kan*) are tested separately. Mesolitica's models are the most
  likely to sound locally natural here. Generic multilingual models may
  sound Indonesian-accented (*estimate*, a common complaint, to be checked
  by listening).

## 7. Automatic emotion detection from plain text (Phase 1.5, not benchmarked yet)
Plan: a small local LLM (e.g. Qwen3 4B/8B Instruct, Apache-2.0) annotates
plain Malay text with the fixed tag vocabulary, sentence by sentence, using
constrained output. Then the same tag→control path runs. To evaluate it,
annotate about 200 Malay sentences by hand and measure tag agreement. The
LLM cost is local and adds about 1–3 s per paragraph (*estimate*).

## 8. Limits of this phase
- **I could not run any model here.** This cloud container has no GPU, and
  its network policy blocks huggingface.co, so no weights could be
  downloaded. The harness is validated end-to-end with a test-only dummy
  adapter (`tests/test_pipeline.py`), not with real speech.
- **Not verified:** MOSS-TTS v1.5's Python API and real VRAM use; the
  Mesolitica checkpoint file names and which F5 config they use (passed as
  adapter args); IndexTTS2's exact licence text (seen only via mirrors).
- **Metric caveats.**
  - Whisper's Malay error rate is not zero even on human speech. Register
    human anchor recordings (`vobench anchors`) to calibrate.
  - ECAPA was trained on VoxCeleb (mostly English).
  - emotion2vec was trained mostly on Chinese and English, so its Malay
    labels are indicative only.
  - UTMOS was trained on English MOS data.
  The **blind listening test decides**. Automatic metrics only screen.

## 9. Decisions needed from you
1. **Commercial or strictly personal use?** This decides whether
   non-commercial models (Malaysian-F5, Fish S2 Pro) can be in the final
   app, or are only comparison points.
2. **The target computer's GPU and VRAM.** Set `vram_mb_max` in
   `benchmark/gates.yaml` to match.
3. **Recordings of the two permanent voices**, with consent. See
   `benchmark/refs/README.md` for the recording list (neutral reference,
   holdout, emotion bank).

## Sources
- VoxCPM2 model card and README (languages, controllable cloning, VRAM, RTF, Apache-2.0): https://huggingface.co/openbmb/VoxCPM2 · https://pypi.org/project/voxcpm
- Chatterbox Multilingual (23 languages incl. `ms`, MIT, exaggeration): https://huggingface.co/ResembleAI/chatterbox/blob/main/README.md · https://pypi.org/project/chatterbox-tts/
- MOSS-TTS family / v1.5 (31 languages incl. Malay, Nano CPU streaming): https://github.com/OpenMOSS/MOSS-TTS · https://ai.gitcode.com/OpenMOSS/MOSS-TTS-v1.5 · https://projects.laion.ai/moss-8b-voice-acting/
- Mesolitica Malaysian TTS models (F5 v1–v3, CC-BY-NC-4.0): https://huggingface.co/mesolitica/Malaysian-F5-TTS-v3 · https://huggingface.co/collections/mesolitica/malaysian-text-to-speech
- Fish Audio S2 Pro (research licence, free-form tags): https://huggingface.co/fishaudio/s2-pro · https://raw.githubusercontent.com/fishaudio/fish-speech/HEAD/README.md
- IndexTTS2 paper (emotion/speaker disentanglement): https://arxiv.org/abs/2506.21619v2 · licence text mirror: https://scancode-licensedb.aboutcode.org/bilibili-model-ula-2025-09-09.html
- Qwen3-TTS (Apache-2.0, 10 languages): https://github.com/QwenLM/Qwen3-TTS · https://arxiv.org/abs/2601.15621
- Seed-VC (GPL-3.0): https://huggingface.co/Plachta/Seed-VC · https://pypi.org/pypi/seed-vc
- Open-source TTS overviews used for screening: https://www.bentoml.com/blog/exploring-the-world-of-open-source-text-to-speech-models · https://www.cartesia.ai/learn/open-source-tts · https://github.com/wildminder/awesome-ai-voice/blob/main/README.md
