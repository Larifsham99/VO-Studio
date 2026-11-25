"""F5-TTS fine-tuned on Malaysian speech by Mesolitica (Malay, Malaysian English,
Mandarin, code-switching).

LICENSE: CC-BY-NC-4.0 (inherits Emilia data terms). Non-commercial only.
Emotion: no tag/instruction control - delivery follows the reference clip, so
this model only does emotion via the per-voice emotion bank.

API verified against f5-tts 1.1.22 (f5_tts/api.py). Checkpoint/vocab file
names in the HF repo could not be verified from this environment; pass them
explicitly via --adapter-arg after downloading, e.g.
  --adapter-arg ckpt_file=/models/Malaysian-F5-TTS-v3/model.safetensors
  --adapter-arg vocab_file=/models/Malaysian-F5-TTS-v3/vocab.txt
  --adapter-arg model_cfg=F5TTS_Base   (match the config the card says it used)
"""

from __future__ import annotations

from ..audio import to_numpy
from .base import SynthesisRequest, TTSAdapter, seed_everything


class F5Malaysian(TTSAdapter):
    name = "f5_malaysian"
    license = "CC-BY-NC-4.0 (weights) / MIT (code)"
    languages = {"ms", "en", "zh"}
    supports_transcript_prompt = True

    def __init__(self, ckpt_file: str = "", vocab_file: str = "", model_cfg: str = "F5TTS_Base"):
        if not ckpt_file or not vocab_file:
            raise ValueError("f5_malaysian needs ckpt_file= and vocab_file= adapter args")
        self.ckpt_file, self.vocab_file, self.model_cfg = ckpt_file, vocab_file, model_cfg

    def load(self, device: str) -> None:
        from f5_tts.api import F5TTS

        self.device = device
        self.model = F5TTS(model=self.model_cfg, ckpt_file=self.ckpt_file,
                           vocab_file=self.vocab_file, device=device)

    def synthesize(self, req: SynthesisRequest):
        if not req.speaker_ref or not req.speaker_ref_text:
            raise ValueError("F5-TTS needs speaker_ref and its transcript")
        seed_everything(req.seed)
        wav, sr, _ = self.model.infer(
            ref_file=req.speaker_ref,
            ref_text=req.speaker_ref_text,
            gen_text=req.text,
            nfe_step=req.params.get("nfe_step", 32),
            cfg_strength=req.params.get("cfg_strength", 2),
            seed=req.seed,
            show_info=lambda *a, **k: None,
        )
        return to_numpy(wav), int(sr)
