"""OpenBMB VoxCPM2 (2B, 30 languages incl. Malay, 48 kHz out, Apache-2.0).

API verified against voxcpm 2.0.3 (voxcpm/core.py + README):
* reference_wav_path           -> timbre only ("isolated reference")
* prompt_wav_path + prompt_text -> continuation: copies timbre AND delivery
* "(instruction)text"          -> controllable cloning / voice design
Vendor-stated: ~8 GB VRAM, RTF ~0.30 on RTX 4090. LoRA fine-tuning supported
(5-10 min of audio) - relevant for the two permanent voices.

Clone modes exercised by the benchmark (req.params["clone_mode"]):
  isolated   - reference_wav_path only (+ instruction for emotion)
  ultimate   - same clip as reference AND continuation prompt (max similarity)
  style_cont - Mode B experiment: identity from speaker_ref (isolated),
               delivery from style_ref (continuation prompt). Whether identity
               leaks from the style prompt is exactly what we measure.
"""

from __future__ import annotations

from ..audio import to_numpy
from ..text.emotion_tags import instruction_for
from .base import SynthesisRequest, TTSAdapter, seed_everything


class VoxCPM2(TTSAdapter):
    name = "voxcpm2"
    license = "Apache-2.0 (code + weights)"
    languages = {"ar", "my", "zh", "da", "nl", "en", "fi", "fr", "de", "el", "he", "hi", "id", "it",
                 "ja", "km", "ko", "lo", "ms", "no", "pl", "pt", "ru", "es", "sw", "sv", "tl", "th",
                 "tr", "vi"}
    supports_instruction = True
    supports_style_ref = True
    supports_transcript_prompt = True

    def __init__(self, model_id: str = "openbmb/VoxCPM2"):
        self.model_id = model_id

    def load(self, device: str) -> None:
        from voxcpm import VoxCPM

        self.device = device
        self.model = VoxCPM.from_pretrained(self.model_id, load_denoiser=False, device=device)

    def synthesize(self, req: SynthesisRequest):
        seed_everything(req.seed)
        mode = req.params.get("clone_mode", "isolated")
        text = req.text
        if req.emotion_strategy == "instruction" and req.tags:
            text = f"({instruction_for(req.tags)}){text}"
        elif req.params.get("instruction"):
            text = f"({req.params['instruction']}){text}"

        kw = dict(cfg_value=req.params.get("cfg_value", 2.0),
                  inference_timesteps=req.params.get("inference_timesteps", 10))
        if req.speaker_ref:
            kw["reference_wav_path"] = req.speaker_ref
        if mode == "ultimate" and req.speaker_ref and req.speaker_ref_text:
            kw["prompt_wav_path"], kw["prompt_text"] = req.speaker_ref, req.speaker_ref_text
        elif mode == "style_cont" and req.style_ref and req.style_ref_text:
            kw["prompt_wav_path"], kw["prompt_text"] = req.style_ref, req.style_ref_text
        wav = self.model.generate(text=text, **kw)
        return to_numpy(wav), int(self.model.tts_model.sample_rate)
