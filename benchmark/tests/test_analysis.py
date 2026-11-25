import numpy as np

from vobench.analysis.style import analyze, style_distance, to_instruction

SR = 16000


def synth(segments, sr=SR):
    """segments: list of (dur_s, f0_start, f0_end, amp); amp 0 = silence.
    Harmonic tone with syllable-rate (5 Hz) amplitude modulation."""
    out = []
    for dur, f0a, f0b, amp in segments:
        n = int(dur * sr)
        if amp == 0:
            out.append(np.zeros(n, dtype=np.float32))
            continue
        f0 = np.linspace(f0a, f0b, n)
        ph = 2 * np.pi * np.cumsum(f0) / sr
        y = sum(np.sin(k * ph) / k for k in range(1, 6))
        env = 0.55 + 0.45 * np.sin(2 * np.pi * 5 * np.arange(n) / sr)
        out.append((amp * env * y / 3).astype(np.float32))
    return np.concatenate(out)


def test_pauses_and_pitch():
    y = synth([(0.3, 0, 0, 0), (1.5, 120, 160, 0.5), (0.6, 0, 0, 0), (1.5, 160, 110, 0.5), (0.3, 0, 0, 0)])
    p = analyze(y, SR)
    assert len(p.pauses) == 1
    assert abs(p.pauses[0].start - 1.8) < 0.1 and abs(p.pauses[0].dur - 0.6) < 0.12
    assert 120 < p.f0_median_hz < 160
    assert p.f0_range_st > 4
    assert p.voiced_ratio > 0.7
    assert p.syllables_per_s is not None and 3.5 < p.syllables_per_s < 6.5  # 5 Hz AM


def test_expressive_vs_flat_distance():
    flat = analyze(synth([(3.0, 140, 142, 0.5)]), SR)
    lively = analyze(synth([(1.0, 120, 220, 0.5), (0.4, 0, 0, 0), (1.0, 220, 110, 0.8)]), SR)
    assert lively.f0_std_st > flat.f0_std_st + 2
    assert "flat" in to_instruction(flat)
    assert style_distance(flat, flat)["mean"] == 0
    assert style_distance(flat, lively)["mean"] > 0.2
