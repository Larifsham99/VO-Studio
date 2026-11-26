"""API tests with a TEST-ONLY dummy model registered inside the test (tones,
not speech). Exercises auth, uploads (phone formats via FFmpeg), Mode A/B
projects, tag handling, chunked generation, WAV/MP3 export."""

import subprocess
import sys
import time
import types

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from vobench import adapters
from vobench.adapters.base import TTSAdapter

CALLS = []


class Dummy(TTSAdapter):
    name = "dummy"
    license = "test"
    languages = {"ms", "en"}
    supports_instruction = True
    supports_style_ref = True

    def load(self, device):
        pass

    def synthesize(self, req):
        CALLS.append(req)
        sr = 16000
        t = np.arange(int(sr * (0.3 + 0.02 * len(req.text)))) / sr
        return (0.3 * np.sin(2 * np.pi * 180 * t)).astype(np.float32), sr


@pytest.fixture
def client(tmp_path, monkeypatch):
    mod = types.ModuleType("latif_test_dummy")
    mod.Dummy = Dummy
    sys.modules["latif_test_dummy"] = mod
    monkeypatch.setitem(adapters.TTS, "dummy", ("latif_test_dummy", "Dummy"))
    from latif_studio import engine as eng_mod
    monkeypatch.setitem(eng_mod.ADAPTER_PACKAGES, "dummy", "numpy")

    refs = tmp_path / "refs"
    refs.mkdir()
    tone = (0.2 * np.sin(np.arange(16000 * 6) / 8)).astype(np.float32)
    sf.write(refs / "a.wav", tone, 16000)
    sf.write(refs / "a_sad.wav", tone, 16000)
    (refs / "manifest.yaml").write_text(
        "voices:\n  - id: suara_a\n    permanent: true\n    consent: 'test'\n    reference: a.wav\n"
        "    transcript: 'satu dua tiga'\n    emotion_bank:\n      sad: {path: a_sad.wav, transcript: 'x'}\n")
    monkeypatch.setenv("LATIF_DATA", str(tmp_path / "data"))
    monkeypatch.setenv("LATIF_MANIFEST", str(refs / "manifest.yaml"))
    monkeypatch.setenv("LATIF_TOKEN", "rahsia")
    monkeypatch.setenv("LATIF_WEB_DIST", str(tmp_path / "nodist"))
    from latif_studio.app import create_app
    from latif_studio.config import Settings

    app = create_app(Settings())
    c = TestClient(app)
    c.headers["Authorization"] = "Bearer rahsia"
    c.tmp = tmp_path
    CALLS.clear()
    return c


def _wait(c, jid, timeout=20):
    t0 = time.time()
    while time.time() - t0 < timeout:
        j = c.get(f"/api/jobs/{jid}").json()
        if j["status"] in ("done", "error"):
            return j
        time.sleep(0.05)
    raise AssertionError("job timeout")


def _phone_recording(path):
    """An .m4a like an Android recorder makes."""
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "sine=frequency=220:duration=8",
                    "-c:a", "aac", str(path)], check=True)


def test_auth_required(client):
    r = client.get("/api/status", headers={"Authorization": "Bearer salah"})
    assert r.status_code == 401
    assert client.get("/api/status").json()["voices"][0]["id"] == "suara_a"


def test_mode_b_with_tags_bank_and_mp3(client):
    p = client.post("/api/projects", json={"name": "Podcast", "mode": "B", "model": "dummy",
                                           "voice_id": "suara_a"}).json()
    j = client.post("/api/generate", json={"project_id": p["id"], "emotion_strategy": "bank",
                                           "text": "[calm] Harga RM25.50 sahaja. [sad] Dia sudah pergi. [oops] ya"}).json()
    assert j["unknown_tags"] == ["oops"]
    done = _wait(client, j["id"])
    assert done["status"] == "done", done
    assert done["chunks"] == 2 and done["has_mp3"]
    texts = [c.text for c in CALLS]
    assert "dua puluh lima ringgit lima puluh sen" in texts[0]  # Malay normaliser applied
    assert all("[" not in t and "oops" not in t for t in texts)  # tags never read aloud
    assert CALLS[1].speaker_ref.endswith("a_sad.wav")  # [sad] used the voice's own sad clip
    a = client.get(f"/api/jobs/{j['id']}/audio?format=mp3&token=rahsia", headers={"Authorization": ""})
    assert a.status_code == 200 and a.headers["content-type"] == "audio/mpeg"


def test_mode_a_upload_requires_consent_and_clones(client):
    rec = client.tmp / "rakaman.m4a"
    _phone_recording(rec)
    bad = client.post("/api/references", data={"name": "Tetamu", "consent": ""},
                      files={"file": ("r.m4a", rec.read_bytes())})
    assert bad.status_code == 400
    ok = client.post("/api/references", data={"name": "Tetamu", "consent": "Ali, lisan, 26/11/2026",
                                              "transcript": "selamat pagi semua"},
                     files={"file": ("r.m4a", rec.read_bytes())})
    assert ok.status_code == 200, ok.text
    ref = ok.json()
    assert 7 < ref["duration_s"] < 9 and "wav" not in ref
    assert ref["profile"] is not None
    p = client.post("/api/projects", json={"name": "Klon", "mode": "A", "model": "dummy",
                                           "reference_id": ref["id"]}).json()
    j = _wait(client, client.post("/api/generate", json={"project_id": p["id"], "text": "[excited] Kita menang!"}).json()["id"])
    assert j["status"] == "done"
    assert CALLS[-1].params["clone_mode"] == "ultimate" and CALLS[-1].tags == ["excited"]
    assert client.delete(f"/api/references/{ref['id']}").status_code == 409  # still used


def test_mode_b_style_route(client):
    rec = client.tmp / "gaya.m4a"
    _phone_recording(rec)
    st = client.post("/api/references", data={"name": "Gaya marah", "consent": "Siti, bertulis",
                                              "transcript": "jangan sentuh barang saya", "purpose": "B"},
                     files={"file": ("g.m4a", rec.read_bytes())}).json()
    p = client.post("/api/projects", json={"name": "B", "mode": "B", "model": "dummy", "voice_id": "suara_a"}).json()
    j = _wait(client, client.post("/api/generate", json={"project_id": p["id"], "text": "Dengar sini.",
                                                         "style_reference_id": st["id"]}).json()["id"])
    assert j["status"] == "done" and j["style_route"] == "native"
    assert CALLS[-1].style_ref and CALLS[-1].speaker_ref.endswith("a.wav")  # identity stays permanent


def test_errors_are_reported(client):
    p = client.post("/api/projects", json={"name": "x", "mode": "B", "model": "tiada_model",
                                           "voice_id": "suara_a"}).json()
    j = _wait(client, client.post("/api/generate", json={"project_id": p["id"], "text": "hai"}).json()["id"])
    assert j["status"] == "error" and "tiada_model" in j["error"]
    assert client.post("/api/projects", json={"name": "x", "mode": "B", "model": "dummy",
                                              "voice_id": "bukan"}).status_code == 400
