"""Resemble AI Chatterbox Multilingual (0.5B, 23 languages incl. Malay 'ms', MIT).

API verified against chatterbox-tts 0.1.7 (chatterbox/mtl_tts.py).
Note: every output carries Resemble's Perth neural watermark (applied inside
generate()). It is inaudible but present; document it to users.
"""

from __future__ import annotations

from ..audio import to_numpy
from ..text.emotion_tags import exaggeration_for
from .base import SynthesisRequest, TTSAdapter, seed_everything


class ChatterboxMultilingual(TTSAdapter):
    name = "chatterbox_ml"
    license = "MIT (code + weights)"
    languages = {"ar", "da", "de", "el", "en", "es", "fi", "fr", "he", "hi", "it", "ja", "ko",
                 "ms", "nl", "no", "pl", "pt", "ru", "sv", "sw", "tr", "zh"}
    supports_exaggeration = True

    def load(self, device: str) -> None:
        from chatterbox.mtl_tts import ChatterboxMultilingualTTS

        self.device = device
        self.model = ChatterboxMultilingualTTS.from_pretrained(device=device)

    def synthesize(self, req: SynthesisRequest):
        seed_everything(req.seed)
        exag = 0.5
        cfg = 0.5
        if req.emotion_strategy == "exaggeration" and req.tags:
            exag = exaggeration_for(req.tags)
            # Resemble's guidance: high exaggeration speeds speech up; lower cfg compensates
            cfg = 0.3 if exag >= 0.7 else 0.5
        exag = req.params.get("exaggeration", exag)
        cfg = req.params.get("cfg_weight", cfg)
        wav = self.model.generate(
            req.text,
            language_id=req.language,
            audio_prompt_path=req.speaker_ref,
            exaggeration=exag,
            cfg_weight=cfg,
            temperature=req.params.get("temperature", 0.8),
        )
        return to_numpy(wav), self.model.sr
