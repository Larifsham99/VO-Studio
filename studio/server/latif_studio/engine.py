"""Generation engine: one worker thread owns the GPU and processes jobs in order.

Text -> tags parsed into control signals (never read aloud) -> chunks ->
model calls -> concatenated WAV (+ MP3 via FFmpeg). No pitch shifting, time
stretching or pre-recorded audio anywhere: every sample comes from the model.
"""

from __future__ import annotations

import importlib.util
import queue
import shutil
import subprocess
import threading
import time
import traceback
from pathlib import Path

from vobench import audio
from vobench.adapters import TTS, get_tts
from vobench.adapters.base import SynthesisRequest, TTSAdapter, pick_device
from vobench.manifest import Manifest, ManifestError, Voice, load_manifest
from vobench.text import malay_normalize
from vobench.text.segment import chunk_text

from .config import Settings
from .store import Store

# python package that must be importable for each adapter
ADAPTER_PACKAGES = {"voxcpm2": "voxcpm", "chatterbox_ml": "chatterbox", "f5_malaysian": "f5_tts"}
MODEL_LABELS = {
    "voxcpm2": "VoxCPM2 (Apache-2.0)",
    "chatterbox_ml": "Chatterbox Multilingual (MIT)",
    "f5_malaysian": "Malaysian F5-TTS (bukan komersial)",
}


def available_models() -> list[dict]:
    out = []
    for name in TTS:
        pkg = ADAPTER_PACKAGES.get(name)
        installed = bool(pkg and importlib.util.find_spec(pkg))
        out.append({"id": name, "label": MODEL_LABELS.get(name, name), "installed": installed})
    return out


def convert_upload(src: Path, dst: Path, sr: int = 24000) -> float:
    """Any phone audio (m4a/webm/ogg/mp3/wav) -> mono WAV. Returns duration."""
    if not shutil.which("ffmpeg"):
        raise RuntimeError("FFmpeg tidak dijumpai di PC. Sila pasang FFmpeg.")
    r = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), "-ac", "1", "-ar", str(sr),
                        "-sample_fmt", "s16", str(dst)], capture_output=True, text=True)
    if r.returncode != 0:
        raise ValueError(f"Fail audio tidak dapat dibaca: {r.stderr.strip()[:200]}")
    y, s = audio.load(dst)
    return audio.duration(y, s)


def _gpu_vram_gb() -> float:
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.get_device_properties(0).total_memory / 2**30
    except ImportError:
        pass
    return 0.0


def transcribe(path: Path, language: str) -> str | None:
    """Optional: only if faster-whisper is installed on the server."""
    if not importlib.util.find_spec("faster_whisper"):
        return None
    from faster_whisper import WhisperModel

    if not hasattr(transcribe, "_m"):
        # Whisper shares the GPU with the TTS model; on small cards (e.g. 6 GB
        # RTX 4050 laptop) keep it on the CPU so the voice model has the VRAM.
        on_gpu = _gpu_vram_gb() >= 10
        transcribe._m = WhisperModel("large-v3-turbo", device="cuda" if on_gpu else "cpu",
                                     compute_type="float16" if on_gpu else "int8")
    segs, _ = transcribe._m.transcribe(str(path), language=language, beam_size=5)
    return " ".join(s.text.strip() for s in segs).strip()


def choose_strategy(adapter: TTSAdapter, requested: str, has_bank: bool) -> str:
    if requested != "auto":
        return requested
    if adapter.supports_instruction:
        return "instruction"
    if adapter.supports_exaggeration:
        return "exaggeration"
    return "bank" if has_bank else "none"


class Engine:
    def __init__(self, settings: Settings, store: Store):
        self.s = settings
        self.store = store
        self.adapter: TTSAdapter | None = None
        self.model_name: str | None = None
        self.state = "idle"  # idle | loading | busy | error
        self.last_error: str | None = None
        self.q: queue.Queue = queue.Queue()
        self.device = pick_device(settings.device)
        self._thread = threading.Thread(target=self._loop, daemon=True, name="latif-gpu-worker")
        self._thread.start()

    # ---- voices -------------------------------------------------------
    def manifest(self) -> Manifest | None:
        if not self.s.manifest.exists():
            return None
        return load_manifest(self.s.manifest)

    def manifest_error(self) -> str | None:
        if not self.s.manifest.exists():
            return f"Fail senarai suara belum ada: {self.s.manifest}"
        try:
            load_manifest(self.s.manifest)
            return None
        except ManifestError as e:
            return str(e)

    def permanent_voices(self) -> list[Voice]:
        try:
            m = self.manifest()
        except ManifestError:
            return []
        return m.permanent if m else []

    # ---- queue ----------------------------------------------------------
    def submit(self, job_id: str) -> None:
        self.q.put(("job", job_id))

    def request_model(self, name: str) -> None:
        self.q.put(("load", name))

    def queue_size(self) -> int:
        return self.q.qsize()

    def _loop(self) -> None:
        while True:
            kind, arg = self.q.get()
            try:
                if kind == "load":
                    self._load(arg)
                else:
                    self._run_job(arg)
            except Exception as e:  # never kill the worker
                self.last_error = f"{type(e).__name__}: {e}"
                if kind == "job":
                    self.store.update("jobs", arg, status="error", error=self.last_error,
                                      trace=traceback.format_exc()[-1500:])
                self.state = "error" if kind == "load" else "idle"

    def _load(self, name: str) -> None:
        if self.model_name == name and self.adapter is not None:
            return
        self.state = "loading"
        if self.adapter is not None:
            self.adapter.unload()
            self.adapter, self.model_name = None, None
        adapter = get_tts(name, **self.s.adapter_args.get(name, {}))
        adapter.load(self.device)
        self.adapter, self.model_name = adapter, name
        self.state = "idle"
        self.last_error = None

    # ---- generation -----------------------------------------------------
    def _run_job(self, job_id: str) -> None:
        job = self.store.get("jobs", job_id)
        if not job or job["status"] != "queued":
            return
        project = self.store.get("projects", job["project_id"])
        if project is None:
            raise ValueError("Projek telah dipadam")
        self.store.update("jobs", job_id, status="running", started=time.time(), progress=0.0)
        self._load(project["model"])
        self.state = "busy"
        try:
            self._generate(job, project)
        finally:
            self.state = "idle"

    def _speaker(self, project: dict) -> tuple[str, str | None, Voice | None]:
        if project["mode"] == "A":
            ref = self.store.get("references", project["reference_id"])
            if ref is None:
                raise ValueError("Rakaman rujukan projek ini sudah dipadam")
            return ref["wav"], ref.get("transcript"), None
        voice = next((v for v in self.permanent_voices() if v.id == project["voice_id"]), None)
        if voice is None:
            raise ValueError(f"Suara tetap '{project['voice_id']}' tidak ada dalam senarai suara")
        return voice.reference.path, voice.reference.transcript, voice

    def _generate(self, job: dict, project: dict) -> None:
        a = self.adapter
        opts = job.get("options", {})
        lang = project.get("language", "ms")
        if a.languages and lang not in a.languages:
            raise ValueError(f"Model {a.name} tidak menyokong bahasa '{lang}'")
        spk_path, spk_text, voice = self._speaker(project)
        strategy = choose_strategy(a, opts.get("emotion_strategy", "auto"), bool(voice and voice.emotion_bank))

        params: dict = {}
        style_ref = style_text = None
        if project["mode"] == "A":
            params["clone_mode"] = "ultimate" if spk_text else "isolated"  # identity + delivery of the reference
        elif opts.get("style_reference_id"):
            st = self.store.get("references", opts["style_reference_id"])
            if st is None:
                raise ValueError("Rakaman gaya sudah dipadam")
            route = opts.get("style_route", "auto")
            if route == "auto":
                route = "native" if (a.supports_style_ref and st.get("transcript")) else "instruction"
            if route == "native":
                if not (a.supports_style_ref and st.get("transcript")):
                    raise ValueError("Model ini tidak menyokong pemindahan gaya terus. Pilih laluan 'arahan'.")
                params["clone_mode"] = "style_cont"
                style_ref, style_text = st["wav"], st["transcript"]
            else:
                if not a.supports_instruction:
                    raise ValueError("Model ini tidak menerima arahan gaya. Cuba VoxCPM2.")
                params["instruction"] = st.get("instruction") or ""
            self.store.update("jobs", job["id"], style_route=route)

        chunks = [c for c in chunk_text(job["text"], self.s.chunk_chars) if c.text]
        if not chunks:
            raise ValueError("Tiada teks untuk dibaca")
        waves, sr, t0 = [], None, time.perf_counter()
        for i, ch in enumerate(chunks):
            text = ch.text
            if opts.get("normalize", True) and lang == "ms":
                text = malay_normalize.normalize(text)
            sp, st_ = spk_path, spk_text
            if strategy == "bank" and voice:
                clip = next((voice.emotion_bank[t] for t in ch.tags if t in voice.emotion_bank), None)
                if clip:
                    sp, st_ = clip.path, clip.transcript
            req = SynthesisRequest(text=text, language=lang, tags=ch.tags, speaker_ref=sp, speaker_ref_text=st_,
                                   style_ref=style_ref, style_ref_text=style_text, emotion_strategy=strategy,
                                   seed=int(opts.get("seed", 1234)) + i, params=dict(params))
            w, sr = a.synthesize(req)
            waves.append(w)
            self.store.update("jobs", job["id"], progress=(i + 1) / len(chunks))

        wav = audio.concat(waves, sr)
        out = self.s.data_dir / "audio" / f"{job['id']}.wav"
        audio.save_wav(out, wav, sr)
        mp3 = None
        if shutil.which("ffmpeg"):
            mp3 = audio.export_mp3(out, out.with_suffix(".mp3"))
        dur = audio.duration(wav, sr)
        wall = time.perf_counter() - t0
        self.store.update("jobs", job["id"], status="done", finished=time.time(), progress=1.0,
                          wav=str(out), mp3=str(mp3) if mp3 else None, duration_s=dur,
                          gen_s=wall, rtf=wall / dur if dur else None, model=a.name,
                          emotion_strategy=strategy, chunks=len(chunks))
