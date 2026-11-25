"""Reference-recording analysis -> structured speaking-style representation.

Everything here is measured from the signal; nothing is guessed. Pitch is
expressed in semitones relative to the speaker's own median, so the profile
describes *how* someone speaks independently of *who* they are. That is what
Mode B needs: transfer the shape, not the absolute pitch of the reference.

Optional inputs improve accuracy:
* transcript    -> syllable count from text instead of acoustic estimate
* emotion_fn    -> callable(wav, sr) -> (label, score) for windowed emotion
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from typing import Callable

import numpy as np

HOP_S = 0.01


@dataclass
class Pause:
    start: float
    end: float

    @property
    def dur(self) -> float:
        return self.end - self.start


@dataclass
class StyleProfile:
    duration_s: float
    speech_s: float
    f0_median_hz: float | None
    f0_std_st: float | None  # pitch variability, semitones
    f0_range_st: float | None  # 5th..95th percentile span, semitones
    f0_slope_st_per_s: float | None  # global declination / rise
    final_rise_ratio: float | None  # share of phrases ending with rising pitch
    voiced_ratio: float  # voiced frames / speech frames (low -> whisper/breathy)
    rms_db_mean: float
    rms_db_range: float  # dynamic range (95th - 5th pct) of speech frames
    syllables_per_s: float | None
    syllable_source: str  # "transcript" | "acoustic"
    pauses: list[Pause] = field(default_factory=list)
    pauses_per_min: float = 0.0
    mean_pause_s: float = 0.0
    emphasis_times: list[float] = field(default_factory=list)
    emotion_timeline: list[dict] = field(default_factory=list)
    f0_contour_st: list[float | None] = field(default_factory=list, repr=False)  # 100 Hz

    def to_json(self, with_contour: bool = False) -> str:
        d = asdict(self)
        if not with_contour:
            d.pop("f0_contour_st")
        return json.dumps(d, indent=2, ensure_ascii=False)

    def summary(self) -> dict:
        d = asdict(self)
        for k in ("f0_contour_st", "pauses", "emphasis_times", "emotion_timeline"):
            d.pop(k)
        return d


def count_syllables_malay(text: str) -> int:
    """Malay orthography is close to phonemic: one vowel nucleus per syllable.
    Word-final diphthongs (pantai, pulau, amboi) count once; other vowel
    sequences are hiatus (di-a, ba-ik, ma-in). Normalise digits beforehand."""
    t = re.sub(r"\[[^\]]*\]", " ", text.lower())
    t = re.sub(r"(ai|au|oi)\b", "A", t)
    return len(re.findall(r"[aeiouA]", t))


def _frames_rms_db(y: np.ndarray, sr: int, hop: int, win: int) -> np.ndarray:
    import librosa

    rms = librosa.feature.rms(y=y, frame_length=win, hop_length=hop, center=True)[0]
    return 20 * np.log10(np.maximum(rms, 1e-6))


def _speech_mask(rms_db: np.ndarray) -> np.ndarray:
    """Energy VAD relative to the loud end of the recording."""
    loud = np.percentile(rms_db, 95)
    floor = np.percentile(rms_db, 10)
    thr = max(floor + 0.35 * (loud - floor), loud - 40)
    mask = rms_db > thr
    # close tiny gaps (<60 ms) so stop consonants don't count as pauses
    return _close_gaps(mask, 6)


def _close_gaps(mask: np.ndarray, max_gap: int) -> np.ndarray:
    m = mask.copy()
    idx = np.flatnonzero(m)
    if idx.size < 2:
        return m
    gaps = np.diff(idx)
    for i, g in enumerate(gaps):
        if 1 < g <= max_gap + 1:
            m[idx[i]: idx[i + 1]] = True
    return m


def _runs(mask: np.ndarray, value: bool) -> list[tuple[int, int]]:
    runs, start = [], None
    for i, v in enumerate(mask):
        if v == value and start is None:
            start = i
        elif v != value and start is not None:
            runs.append((start, i))
            start = None
    if start is not None:
        runs.append((start, len(mask)))
    return runs


def _acoustic_syllables(rms_db: np.ndarray, speech: np.ndarray) -> int:
    """Syllable nuclei = prominent energy peaks inside speech (de Jong & Wempe style)."""
    from scipy.signal import find_peaks

    e = np.convolve(rms_db, np.ones(5) / 5, mode="same")
    peaks, _ = find_peaks(e, prominence=2.0, distance=8)  # >= 80 ms apart
    return int(np.sum(speech[peaks]))


def analyze(
    y: np.ndarray,
    sr: int,
    transcript: str | None = None,
    emotion_fn: Callable[[np.ndarray, int], tuple[str, float]] | None = None,
    emotion_window_s: float = 3.0,
    min_pause_s: float = 0.2,
) -> StyleProfile:
    import librosa

    y = np.asarray(y, dtype=np.float32)
    hop = int(sr * HOP_S)
    win = 4 * hop
    dur = len(y) / sr
    rms_db = _frames_rms_db(y, sr, hop, win)
    speech = _speech_mask(rms_db)

    f0, voiced, _ = librosa.pyin(y, fmin=60, fmax=700, sr=sr, frame_length=win * 2, hop_length=hop)
    n = min(len(f0), len(speech))
    f0, voiced, speech, rms_db = f0[:n], voiced[:n], speech[:n], rms_db[:n]
    voiced = voiced & speech & ~np.isnan(f0)

    # pauses = non-speech runs inside the utterance (ignore leading/trailing silence)
    sp_idx = np.flatnonzero(speech)
    pauses: list[Pause] = []
    if sp_idx.size:
        inner = speech[sp_idx[0]: sp_idx[-1] + 1]
        for a, b in _runs(inner, False):
            if (b - a) * HOP_S >= min_pause_s:
                pauses.append(Pause(round((a + sp_idx[0]) * HOP_S, 3), round((b + sp_idx[0]) * HOP_S, 3)))
    speech_s = float(speech.sum() * HOP_S)

    f0_med = f0_std = f0_rng = slope = final_rise = None
    contour: list[float | None] = [None] * n
    if voiced.sum() >= 10:
        med = float(np.median(f0[voiced]))
        st = 12 * np.log2(f0 / med)
        f0_med = med
        f0_std = float(np.std(st[voiced]))
        f0_rng = float(np.percentile(st[voiced], 95) - np.percentile(st[voiced], 5))
        t = np.arange(n) * HOP_S
        slope = float(np.polyfit(t[voiced], st[voiced], 1)[0])
        contour = [round(float(v), 2) if ok else None for v, ok in zip(st, voiced)]
        # phrase-final movement: compare last 150 ms vs preceding 300 ms of voiced pitch
        rises = total = 0
        for a, b in _runs(speech, True):
            seg = np.flatnonzero(voiced[a:b]) + a
            if seg.size < 20:
                continue
            tail, head = st[seg[-15:]], st[seg[-45:-15]]
            total += 1
            rises += int(np.median(tail) - np.median(head) > 1.0)
        final_rise = rises / total if total else None

    sp_db = rms_db[speech] if speech.any() else rms_db
    if transcript:
        syl, syl_src = count_syllables_malay(transcript), "transcript"
    else:
        syl, syl_src = _acoustic_syllables(rms_db, speech), "acoustic"
    rate = syl / speech_s if speech_s > 0.3 else None

    # emphasis: frames where both loudness and pitch are well above the speaker's norm
    emphasis: list[float] = []
    if f0_med is not None:
        st = 12 * np.log2(np.where(voiced, f0, f0_med) / f0_med)
        ez = (rms_db - sp_db.mean()) / (sp_db.std() + 1e-6)
        pz = (st - np.mean(st[voiced])) / (np.std(st[voiced]) + 1e-6)
        hot = voiced & (ez > 1.0) & (pz > 1.0)
        emphasis = [round(((a + b) / 2) * HOP_S, 2) for a, b in _runs(hot, True) if b - a >= 5]

    timeline: list[dict] = []
    if emotion_fn is not None:
        step = int(emotion_window_s * sr)
        for s in range(0, len(y), step):
            w = y[s: s + step]
            if len(w) < sr:
                break
            label, score = emotion_fn(w, sr)
            timeline.append({"start": round(s / sr, 2), "end": round((s + len(w)) / sr, 2),
                             "label": label, "score": round(float(score), 3)})

    return StyleProfile(
        duration_s=round(dur, 3),
        speech_s=round(speech_s, 3),
        f0_median_hz=f0_med,
        f0_std_st=f0_std,
        f0_range_st=f0_rng,
        f0_slope_st_per_s=slope,
        final_rise_ratio=final_rise,
        voiced_ratio=float(voiced.sum() / max(speech.sum(), 1)),
        rms_db_mean=float(sp_db.mean()),
        rms_db_range=float(np.percentile(sp_db, 95) - np.percentile(sp_db, 5)),
        syllables_per_s=rate,
        syllable_source=syl_src,
        pauses=pauses,
        pauses_per_min=len(pauses) / (dur / 60) if dur > 0 else 0.0,
        mean_pause_s=float(np.mean([p.dur for p in pauses])) if pauses else 0.0,
        emphasis_times=emphasis,
        emotion_timeline=timeline,
        f0_contour_st=contour,
    )


def to_instruction(p: StyleProfile) -> str:
    """Render a profile as a style instruction for instruct-capable TTS.

    Thresholds are starting points for Malay read/conversational speech and
    are meant to be re-calibrated on the user's own recordings.
    """
    parts: list[str] = []
    if p.syllables_per_s is not None:
        if p.syllables_per_s > 6.5:
            parts.append("fast pace")
        elif p.syllables_per_s < 4.0:
            parts.append("slow, deliberate pace")
    if p.f0_std_st is not None:
        if p.f0_std_st > 4.0:
            parts.append("very expressive, wide pitch movement")
        elif p.f0_std_st < 2.0:
            parts.append("flat, even intonation")
    if p.voiced_ratio < 0.35:
        parts.append("whispering, breathy")
    if p.pauses_per_min > 20:
        parts.append("frequent dramatic pauses")
    if p.final_rise_ratio is not None and p.final_rise_ratio > 0.5:
        parts.append("rising, questioning phrase endings")
    if p.emotion_timeline:
        labels = [e["label"] for e in p.emotion_timeline if e["label"] not in ("neutral", "other", "<unk>")]
        if labels:
            dom = max(set(labels), key=labels.count)
            parts.append(f"{dom} tone")
    return ", ".join(parts)


def style_distance(a: StyleProfile, b: StyleProfile) -> dict:
    """Text-independent comparison of two profiles (for Mode B with new text)."""

    def rel(x, y):
        if x is None or y is None or max(abs(x), abs(y)) < 1e-9:
            return None
        return abs(x - y) / max(abs(x), abs(y))

    d = {
        "rate_rel_diff": rel(a.syllables_per_s, b.syllables_per_s),
        "f0_std_rel_diff": rel(a.f0_std_st, b.f0_std_st),
        "f0_range_rel_diff": rel(a.f0_range_st, b.f0_range_st),
        "dyn_range_rel_diff": rel(a.rms_db_range, b.rms_db_range),
        "pause_rate_rel_diff": rel(a.pauses_per_min, b.pauses_per_min),
        "voiced_ratio_abs_diff": abs(a.voiced_ratio - b.voiced_ratio),
    }
    vals = [v for v in d.values() if v is not None]
    d["mean"] = float(np.mean(vals)) if vals else None
    return d
