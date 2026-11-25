"""Lazy-loaded evaluation models. Each returns None when its dependency is
missing so scoring degrades to "not measured" (null), never to a fake value.

These run in the separate *eval* venv; none of them ship in the final app.
  ASR      faster-whisper large-v3        (MIT)
  Speaker  SpeechBrain ECAPA-TDNN          (Apache-2.0)
  MOS      UTMOS22 strong via SpeechMOS    (MIT)  - predicts naturalness 1..5
  Emotion  emotion2vec_plus_large (FunASR) - 9-class speech emotion
"""

from __future__ import annotations

import warnings
from functools import lru_cache

import numpy as np

from .. import audio


def _device() -> str:
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


@lru_cache(maxsize=None)
def whisper(model_size: str = "large-v3"):
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        warnings.warn("faster-whisper not installed: WER/CER not measured")
        return None
    dev = _device()
    return WhisperModel(model_size, device=dev, compute_type="float16" if dev == "cuda" else "int8")


def transcribe(path: str, language: str) -> str | None:
    m = whisper()
    if m is None:
        return None
    segs, _ = m.transcribe(path, language=language, beam_size=5, condition_on_previous_text=False,
                           vad_filter=False)
    return " ".join(s.text.strip() for s in segs).strip()


@lru_cache(maxsize=None)
def ecapa():
    try:
        try:
            from speechbrain.inference.speaker import EncoderClassifier
        except ImportError:
            from speechbrain.pretrained import EncoderClassifier
    except ImportError:
        warnings.warn("speechbrain not installed: speaker similarity not measured")
        return None
    return EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb",
                                          run_opts={"device": _device()})


@lru_cache(maxsize=4096)
def speaker_embedding(path: str):
    m = ecapa()
    if m is None:
        return None
    import torch

    y, _ = audio.load(path, sr=16000)
    with torch.no_grad():
        e = m.encode_batch(torch.from_numpy(y)[None].to(m.device)).squeeze().cpu().numpy()
    return e / (np.linalg.norm(e) + 1e-9)


def speaker_sim(a: str | None, b: str | None) -> float | None:
    if not a or not b:
        return None
    ea, eb = speaker_embedding(a), speaker_embedding(b)
    if ea is None or eb is None:
        return None
    return float(np.dot(ea, eb))


@lru_cache(maxsize=None)
def utmos():
    try:
        import torch

        return torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True).to(_device()).eval()
    except Exception as e:
        warnings.warn(f"UTMOS unavailable ({e}): MOS not measured")
        return None


def mos(path: str) -> float | None:
    m = utmos()
    if m is None:
        return None
    import torch

    y, sr = audio.load(path, sr=16000)
    with torch.no_grad():
        return float(m(torch.from_numpy(y)[None].to(next(m.parameters()).device), sr).item())


@lru_cache(maxsize=None)
def emotion2vec():
    try:
        from funasr import AutoModel
    except ImportError:
        warnings.warn("funasr not installed: speech emotion not measured")
        return None
    return AutoModel(model="emotion2vec/emotion2vec_plus_large", hub="hf", disable_update=True)


def emotion(path: str) -> dict | None:
    m = emotion2vec()
    if m is None:
        return None
    res = m.generate(path, granularity="utterance", extract_embedding=False)[0]
    labels = [lbl.split("/")[-1] for lbl in res["labels"]]
    scores = [float(s) for s in res["scores"]]
    i = int(np.argmax(scores))
    return {"label": labels[i], "score": scores[i], "all": dict(zip(labels, scores))}
