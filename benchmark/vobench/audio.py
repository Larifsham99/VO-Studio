"""Small audio helpers: load/save/resample/concat/export (FFmpeg for MP3)."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf


def to_numpy(wav) -> np.ndarray:
    """Accept torch tensors / lists / arrays, return float32 mono 1-D."""
    if hasattr(wav, "detach"):
        wav = wav.detach().cpu().float().numpy()
    a = np.asarray(wav, dtype=np.float32)
    if a.ndim == 2:
        # (channels, n) or (n, channels)
        a = a.mean(axis=0) if a.shape[0] <= 2 else a.mean(axis=1)
    return a.reshape(-1)


def resample(a: np.ndarray, sr: int, target_sr: int) -> np.ndarray:
    if sr == target_sr:
        return a
    try:
        import librosa

        return librosa.resample(a, orig_sr=sr, target_sr=target_sr).astype(np.float32)
    except ImportError:
        from scipy.signal import resample_poly
        from math import gcd

        g = gcd(sr, target_sr)
        return resample_poly(a, target_sr // g, sr // g).astype(np.float32)


def load(path: str | Path, sr: int | None = None) -> tuple[np.ndarray, int]:
    a, file_sr = sf.read(str(path), dtype="float32", always_2d=False)
    a = to_numpy(a.T if a.ndim == 2 else a)
    if sr and sr != file_sr:
        return resample(a, file_sr, sr), sr
    return a, file_sr


def save_wav(path: str | Path, a: np.ndarray, sr: int) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    peak = float(np.max(np.abs(a))) if a.size else 0.0
    if peak > 1.0:  # avoid clipping; report it instead of hiding it
        a = a / peak
    sf.write(str(path), a, sr, subtype="PCM_16")
    return path


def export_mp3(wav_path: str | Path, mp3_path: str | Path, bitrate: str = "192k") -> Path:
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg not found on PATH")
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(wav_path), "-codec:a", "libmp3lame",
         "-b:a", bitrate, str(mp3_path)],
        check=True,
    )
    return Path(mp3_path)


def concat(chunks: list[np.ndarray], sr: int, gap_s: float = 0.18, crossfade_s: float = 0.0) -> np.ndarray:
    """Join long-form chunks with a short natural pause (no time-stretching)."""
    if not chunks:
        return np.zeros(0, dtype=np.float32)
    gap = np.zeros(int(gap_s * sr), dtype=np.float32)
    out = [chunks[0]]
    for c in chunks[1:]:
        if crossfade_s > 0 and gap_s == 0:
            n = min(int(crossfade_s * sr), len(out[-1]), len(c))
            fade = np.linspace(0, 1, n, dtype=np.float32)
            prev = out[-1]
            prev[-n:] = prev[-n:] * (1 - fade) + c[:n] * fade
            out.append(c[n:])
        else:
            out += [gap, c]
    return np.concatenate(out).astype(np.float32)


def duration(a: np.ndarray, sr: int) -> float:
    return len(a) / float(sr) if sr else 0.0
