"""Voice-conversion back-ends for Mode B pipelines.

Mode B = keep a permanent voice's identity, take delivery from a reference.
Two VC-based routes are benchmarked:
  s2s        reference recording --VC--> permanent voice (same words as ref;
             prosody is preserved exactly because it IS the reference's prosody)
  tts_then_vc  TTS clones the *style reference* speaker saying new text
             (delivery follows reference), then VC swaps timbre to the
             permanent voice. Works for arbitrary text.

Seed-VC v2 (GPL-3.0): API verified against seed-vc 0.4.3 (seed_vc/api.py).
  convert_style=False keeps the source's accent/emotion (what Mode B wants).
Chatterbox VC (MIT): verified against chatterbox-tts 0.1.7 (chatterbox/vc.py).
"""

from __future__ import annotations

import numpy as np
import soundfile as sf

from ..audio import to_numpy
from .base import VCAdapter


class SeedVC2(VCAdapter):
    name = "seed_vc2"
    license = "GPL-3.0 (code + weights)"

    def load(self, device: str) -> None:
        import seed_vc.api  # noqa: F401  (models load lazily on first call)

        self.device = device

    @staticmethod
    def _audio(path: str):
        from seed_vc.Models.audio import AudioData

        a, sr = sf.read(path, dtype="int16", always_2d=False)
        if a.ndim == 2:
            a = a.mean(axis=1).astype(np.int16)
        return AudioData(samples=a, mel_chunks=None, duration=len(a) / sr,
                         samples_count=len(a), sample_rate=sr, metadata=None)

    def convert(self, source_path: str, target_ref_path: str, **kw):
        from seed_vc.api import inference_v2

        sr, wav = inference_v2(
            source=self._audio(source_path),
            target=self._audio(target_ref_path),
            diffusion_steps=kw.get("diffusion_steps", 30),
            similarity_cfg_rate=kw.get("similarity_cfg_rate", 0.7),
            intelligibility_cfg_rate=kw.get("intelligibility_cfg_rate", 0.7),
            convert_style=kw.get("convert_style", False),
        )
        return to_numpy(wav), int(sr)


class ChatterboxVC(VCAdapter):
    name = "chatterbox_vc"
    license = "MIT (code + weights)"

    def load(self, device: str) -> None:
        from chatterbox.vc import ChatterboxVC as _VC

        self.device = device
        self.model = _VC.from_pretrained(device)

    def convert(self, source_path: str, target_ref_path: str, **kw):
        wav = self.model.generate(source_path, target_voice_path=target_ref_path)
        return to_numpy(wav), self.model.sr
