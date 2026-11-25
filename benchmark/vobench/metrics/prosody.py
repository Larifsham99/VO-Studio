"""Prosody metrics built on analysis.style (signal-measured, model-independent)."""

from __future__ import annotations

import numpy as np

from ..analysis.style import HOP_S, StyleProfile
from ..text.emotion_tags import EMOTIONS


def contour_correlation(ya: np.ndarray, yb: np.ndarray, sr: int, pa: StyleProfile, pb: StyleProfile) -> float | None:
    """Pitch-contour similarity for SAME-TEXT pairs: DTW-align on MFCCs, then
    Pearson correlation of speaker-normalised (semitone) F0 on frames voiced
    in both. 1.0 = identical intonation shape."""
    import librosa

    hop = int(sr * HOP_S)
    ma = librosa.feature.mfcc(y=ya, sr=sr, n_mfcc=20, hop_length=hop)
    mb = librosa.feature.mfcc(y=yb, sr=sr, n_mfcc=20, hop_length=hop)
    _, wp = librosa.sequence.dtw(X=ma, Y=mb, metric="cosine")
    fa, fb = pa.f0_contour_st, pb.f0_contour_st
    xs, ys = [], []
    for i, j in wp[::-1]:
        if i < len(fa) and j < len(fb) and fa[i] is not None and fb[j] is not None:
            xs.append(fa[i])
            ys.append(fb[j])
    if len(xs) < 30:
        return None
    return float(np.corrcoef(xs, ys)[0, 1])


def delivery_checks(tag: str, gen: StyleProfile, base: StyleProfile) -> dict:
    """Did the tagged output move away from the neutral baseline (same voice,
    same words) in the direction the tag implies? Directional, not absolute:
    it catches "tag ignored" and "tag read aloud" failures cheaply; the
    listening test judges whether it is *convincing*."""
    out: dict = {}
    d_rms = gen.rms_db_mean - base.rms_db_mean
    d_f0 = (gen.f0_std_st or 0) - (base.f0_std_st or 0)
    d_rate = ((gen.syllables_per_s or 0) - (base.syllables_per_s or 0))
    out.update(d_rms_db=d_rms, d_f0_std_st=d_f0, d_rate_syl_s=d_rate,
               d_voiced_ratio=gen.voiced_ratio - base.voiced_ratio)
    spec = EMOTIONS.get(tag)
    if spec is None:
        return out
    if tag == "whispers":
        ok = gen.voiced_ratio < base.voiced_ratio - 0.2 and d_rms < -3
    elif tag == "shouts":
        ok = d_rms > 4 or (gen.f0_median_hz or 0) > (base.f0_median_hz or 1e9) * 1.15
    elif tag in ("softly",):
        ok = d_rms < -2
    elif spec.arousal >= 0.7:
        ok = d_rms > 1.5 or d_f0 > 0.7 or d_rate > 0.4
    elif spec.arousal <= 0.3:
        ok = d_rms < -1.5 or d_rate < -0.4 or d_f0 < -0.5
    else:
        ok = None  # mid-arousal tags (curious, sarcastic...) -> listening test only
    out["direction_ok"] = ok
    return out
