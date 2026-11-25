"""End-to-end plumbing: run -> score -> report with a TEST-ONLY dummy adapter.
The dummy produces tones, not speech; it exists to exercise the harness, and is
registered only inside this test."""
import json
from pathlib import Path

import numpy as np
import soundfile as sf

from vobench import adapters
from vobench.adapters.base import TTSAdapter
from vobench.manifest import load_manifest, load_suite
from vobench.report import build_report
from vobench.run import run
from vobench.score import score_all


class Dummy(TTSAdapter):
    name = "dummy"
    license = "test"
    languages = {"ms", "en"}
    supports_instruction = True

    def load(self, device):
        pass

    def synthesize(self, req):
        sr = 16000
        n = int(sr * (0.5 + 0.03 * len(req.text)))
        t = np.arange(n) / sr
        return (0.3 * np.sin(2 * np.pi * 150 * t)).astype(np.float32), sr


def test_end_to_end(tmp_path, monkeypatch):
    import sys, types
    mod = types.ModuleType("vobench_test_dummy")
    mod.Dummy = Dummy
    sys.modules["vobench_test_dummy"] = mod
    monkeypatch.setitem(adapters.TTS, "dummy", ("vobench_test_dummy", "Dummy"))

    refs = tmp_path / "refs"
    refs.mkdir()
    for name in ("a.wav", "a_hold.wav", "style.wav"):
        sf.write(refs / name, (0.2 * np.sin(np.arange(32000) / 10)).astype(np.float32), 16000)
    (refs / "manifest.yaml").write_text(
        "voices:\n"
        "  - {id: va, permanent: true, consent: 'test', reference: a.wav, transcript: 'satu dua', holdout: a_hold.wav}\n"
        "  - {id: nocons, permanent: false, consent: 'x', reference: a.wav, transcript: 'x'}\n"
        "style_refs:\n  - {id: s1, consent: 'test', path: style.wav, transcript: 'tiga empat'}\n")
    manifest = load_manifest(refs / "manifest.yaml")
    suites = [load_suite(Path("suites") / f"{n}.yaml") for n in ("malay", "emotion", "longform", "style_transfer")]
    out = run("dummy", suites, manifest, tmp_path / "results", "cpu", {}, {"instruction", "tts_then_vc"}, limit=6)
    recs = [json.loads(line) for line in (out / "records.jsonl").read_text().splitlines()]
    assert recs and all(r["status"] in ("ok", "needs_vc", "unsupported_language") for r in recs), \
        [r.get("error") for r in recs]
    assert any(r.get("chunks") for r in recs)
    score_all(tmp_path / "results")
    rep = build_report(tmp_path / "results", Path("gates.yaml"))
    assert "Viability gates" in rep.read_text()
    assert (tmp_path / "results" / "listening.html").exists()


def test_consent_required(tmp_path):
    import pytest
    from vobench.manifest import ManifestError
    sf.write(tmp_path / "a.wav", np.zeros(16000, dtype=np.float32), 16000)
    (tmp_path / "m.yaml").write_text("voices:\n  - {id: v, reference: a.wav}\n")
    with pytest.raises(ManifestError, match="consent"):
        load_manifest(tmp_path / "m.yaml")
