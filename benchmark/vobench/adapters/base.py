"""Model adapter interface. Every candidate model is wrapped behind this so the
benchmark (and later the app) can swap models without touching anything else."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class SynthesisRequest:
    text: str  # clean text, tags already removed
    language: str = "ms"  # ISO-639-1
    tags: list[str] = field(default_factory=list)  # active emotion tags (canonical names)
    speaker_ref: str | None = None  # timbre reference (permanent voice, or Mode A reference)
    speaker_ref_text: str | None = None  # its transcript (needed by continuation models)
    style_ref: str | None = None  # Mode B prosody/emotion reference
    style_ref_text: str | None = None
    emotion_strategy: str = "instruction"  # instruction | exaggeration | bank | none
    seed: int = 1234
    params: dict = field(default_factory=dict)  # adapter-specific overrides


class TTSAdapter:
    name: str = "base"
    license: str = "unknown"
    languages: set[str] = set()
    # capability flags used by the runner to decide which strategies to try
    supports_clone = True
    supports_instruction = False  # natural-language emotion/style control
    supports_exaggeration = False
    supports_style_ref = False  # separate prosody reference, identity from speaker_ref
    supports_transcript_prompt = False  # continuation cloning needs ref transcript

    def load(self, device: str) -> None:
        raise NotImplementedError

    def synthesize(self, req: SynthesisRequest) -> tuple[np.ndarray, int]:
        raise NotImplementedError

    def unload(self) -> None:
        import gc

        for k in list(vars(self)):
            if k not in ("device",):
                setattr(self, k, None)
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass


class VCAdapter:
    """Voice conversion: keep source content+prosody, replace timbre."""

    name: str = "base_vc"
    license: str = "unknown"

    def load(self, device: str) -> None:
        raise NotImplementedError

    def convert(self, source_path: str, target_ref_path: str, **kw) -> tuple[np.ndarray, int]:
        raise NotImplementedError


def seed_everything(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def pick_device(requested: str | None) -> str:
    if requested:
        return requested
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"
