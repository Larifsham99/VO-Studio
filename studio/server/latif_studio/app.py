"""HTTP API + serves the phone web app. Every /api route needs the access token."""

from __future__ import annotations

import secrets
import shutil
import tempfile
from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from vobench.profiling import system_info
from vobench.text.emotion_tags import EMOTIONS, parse

from .config import Settings
from .engine import Engine, available_models, convert_upload, transcribe
from .store import Store


class ProjectIn(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    mode: str = Field(pattern="^[AB]$")
    model: str
    language: str = "ms"
    voice_id: str | None = None  # Mode B
    reference_id: str | None = None  # Mode A


class GenerateIn(BaseModel):
    project_id: str
    text: str = Field(min_length=1, max_length=20000)
    emotion_strategy: str = Field("auto", pattern="^(auto|instruction|exaggeration|bank|none)$")
    normalize: bool = True
    style_reference_id: str | None = None  # Mode B only
    style_route: str = Field("auto", pattern="^(auto|native|instruction)$")
    seed: int = 1234


def create_app(settings: Settings | None = None, engine: Engine | None = None) -> FastAPI:
    s = settings or Settings()
    store = engine.store if engine else Store(s.data_dir)
    eng = engine or Engine(s, store)
    app = FastAPI(title="Latif Voice AI", docs_url=None, redoc_url=None)
    app.state.engine, app.state.settings = eng, s

    def auth(request: Request, token: str | None = Query(None)) -> None:
        h = request.headers.get("authorization", "")
        given = h[7:] if h.lower().startswith("bearer ") else (token or "")
        if not secrets.compare_digest(given.encode(), s.token.encode()):
            raise HTTPException(401, "Token salah atau tiada")

    api = Depends(auth)

    def public_job(j: dict) -> dict:
        return {k: v for k, v in j.items() if k not in ("wav", "mp3", "trace")} | {
            "has_mp3": bool(j.get("mp3"))}

    def public_ref(r: dict) -> dict:
        return {k: v for k, v in r.items() if k != "wav"}

    # ---------------- status / setup ----------------
    @app.get("/api/status", dependencies=[api])
    def status():
        voices = [{"id": v.id, "has_emotion_bank": sorted(v.emotion_bank)} for v in eng.permanent_voices()]
        return {"state": eng.state, "model": eng.model_name, "device": eng.device, "queue": eng.queue_size(),
                "last_error": eng.last_error, "models": available_models(), "voices": voices,
                "manifest_error": eng.manifest_error(), "system": system_info(),
                "emotions": [{"id": k, "group": v.group} for k, v in EMOTIONS.items()]}

    @app.post("/api/model/{name}", dependencies=[api])
    def load_model(name: str):
        if name not in {m["id"] for m in available_models() if m["installed"]}:
            raise HTTPException(400, f"Model '{name}' tidak dipasang pada server ini")
        eng.request_model(name)
        return {"queued": True}

    # ---------------- references (uploads) ----------------
    @app.post("/api/references", dependencies=[api])
    def add_reference(file: UploadFile = File(...), name: str = Form(...), consent: str = Form(""),
                      transcript: str = Form(""), language: str = Form("ms"), purpose: str = Form("A")):
        if len(consent.strip()) < 5:
            raise HTTPException(400, "Sila nyatakan siapa memberi kebenaran untuk rakaman ini")
        with tempfile.TemporaryDirectory() as td:
            raw = Path(td) / "upload"
            with raw.open("wb") as f:
                shutil.copyfileobj(file.file, f, length=1 << 20)
            if raw.stat().st_size > s.max_upload_mb * 2**20:
                raise HTTPException(413, f"Fail melebihi {s.max_upload_mb} MB")
            ref = store.add("references", {"name": name.strip()[:80], "consent": consent.strip()[:300],
                                           "language": language, "purpose": purpose})
            dst = s.data_dir / "refs" / f"{ref['id']}.wav"
            try:
                dur = convert_upload(raw, dst)
            except (ValueError, RuntimeError) as e:
                store.delete("references", ref["id"])
                raise HTTPException(400, str(e))
        if dur < 3 or dur > 60:
            dst.unlink(missing_ok=True)
            store.delete("references", ref["id"])
            raise HTTPException(400, f"Rakaman {dur:.1f} s. Perlu antara 3 hingga 60 saat (terbaik 10-20 s).")
        text = transcript.strip()
        auto = False
        if not text:
            text = transcribe(dst, language) or ""
            auto = bool(text)
        profile, instruction = None, None
        try:
            from vobench import audio
            from vobench.analysis.style import analyze, to_instruction

            y, sr = audio.load(dst, sr=16000)
            p = analyze(y, sr, transcript=text or None)
            profile, instruction = p.summary(), to_instruction(p)
            profile["pauses"] = len(p.pauses)
            profile["emphasis_points"] = len(p.emphasis_times)
        except ImportError:
            pass  # librosa missing -> no analysis, still usable for cloning
        ref = store.update("references", ref["id"], wav=str(dst), duration_s=dur, transcript=text,
                           transcript_auto=auto, profile=profile, instruction=instruction)
        return public_ref(ref)

    @app.get("/api/references", dependencies=[api])
    def list_references():
        return [public_ref(r) for r in store.list("references")]

    @app.get("/api/references/{rid}/audio", dependencies=[api])
    def reference_audio(rid: str):
        r = store.get("references", rid)
        if not r or not r.get("wav"):
            raise HTTPException(404)
        return FileResponse(r["wav"], media_type="audio/wav")

    @app.patch("/api/references/{rid}", dependencies=[api])
    def edit_reference(rid: str, body: dict):
        if not store.get("references", rid):
            raise HTTPException(404)
        allowed = {k: str(v)[:2000] for k, v in body.items() if k in ("name", "transcript")}
        return public_ref(store.update("references", rid, **allowed, transcript_auto=False))

    @app.delete("/api/references/{rid}", dependencies=[api])
    def delete_reference(rid: str):
        if any(p.get("reference_id") == rid for p in store.list("projects")):
            raise HTTPException(409, "Rakaman ini digunakan oleh projek. Padam projek itu dahulu.")
        r = store.delete("references", rid)
        if not r:
            raise HTTPException(404)
        if r.get("wav"):
            Path(r["wav"]).unlink(missing_ok=True)
        return {"deleted": rid}

    # ---------------- projects ----------------
    @app.post("/api/projects", dependencies=[api])
    def create_project(p: ProjectIn):
        if p.mode == "A":
            r = store.get("references", p.reference_id or "")
            if not r:
                raise HTTPException(400, "Mod A perlukan rakaman rujukan yang dibenarkan")
        else:
            if p.voice_id not in {v.id for v in eng.permanent_voices()}:
                raise HTTPException(400, "Mod B perlukan salah satu suara tetap")
        return store.add("projects", p.model_dump())

    @app.get("/api/projects", dependencies=[api])
    def list_projects():
        return store.list("projects")

    @app.delete("/api/projects/{pid}", dependencies=[api])
    def delete_project(pid: str):
        if not store.delete("projects", pid):
            raise HTTPException(404)
        for j in store.list("jobs", project_id=pid):
            for k in ("wav", "mp3"):
                if j.get(k):
                    Path(j[k]).unlink(missing_ok=True)
            store.delete("jobs", j["id"])
        return {"deleted": pid}

    # ---------------- generation ----------------
    @app.post("/api/generate", dependencies=[api])
    def generate(g: GenerateIn):
        proj = store.get("projects", g.project_id)
        if not proj:
            raise HTTPException(404, "Projek tidak dijumpai")
        if g.style_reference_id and proj["mode"] != "B":
            raise HTTPException(400, "Rakaman gaya hanya untuk Mod B")
        pr = parse(g.text)
        if not pr.plain_text:
            raise HTTPException(400, "Tiada teks untuk dibaca")
        job = store.add("jobs", {"project_id": g.project_id, "text": g.text, "status": "queued", "progress": 0.0,
                                 "unknown_tags": pr.unknown_tags,
                                 "options": g.model_dump(exclude={"project_id", "text"})})
        eng.submit(job["id"])
        return public_job(job)

    @app.get("/api/jobs", dependencies=[api])
    def list_jobs(project_id: str | None = None, limit: int = 50):
        rows = store.list("jobs", project_id=project_id) if project_id else store.list("jobs")
        return [public_job(j) for j in rows[:limit]]

    @app.get("/api/jobs/{jid}", dependencies=[api])
    def get_job(jid: str):
        j = store.get("jobs", jid)
        if not j:
            raise HTTPException(404)
        return public_job(j)

    @app.get("/api/jobs/{jid}/audio", dependencies=[api])
    def job_audio(jid: str, format: str = "wav", download: bool = False):
        j = store.get("jobs", jid)
        if not j or j.get("status") != "done":
            raise HTTPException(404, "Audio belum siap")
        path = j.get("mp3") if format == "mp3" else j.get("wav")
        if not path or not Path(path).exists():
            raise HTTPException(404, f"Format {format} tiada (FFmpeg dipasang?)")
        name = f"latif_{jid}.{format}" if download else None
        return FileResponse(path, media_type="audio/mpeg" if format == "mp3" else "audio/wav", filename=name)

    @app.delete("/api/jobs/{jid}", dependencies=[api])
    def delete_job(jid: str):
        j = store.delete("jobs", jid)
        if not j:
            raise HTTPException(404)
        for k in ("wav", "mp3"):
            if j.get(k):
                Path(j[k]).unlink(missing_ok=True)
        return {"deleted": jid}

    # ---------------- phone web app ----------------
    if (s.web_dist / "index.html").exists():
        app.mount("/assets", StaticFiles(directory=s.web_dist / "assets"), name="assets")

        @app.get("/{path:path}", include_in_schema=False)
        def spa(path: str):
            f = (s.web_dist / path).resolve()
            if path and f.is_file() and s.web_dist.resolve() in f.parents:
                return FileResponse(f)
            return FileResponse(s.web_dist / "index.html")
    else:
        @app.get("/", include_in_schema=False)
        def no_web():
            return HTMLResponse("<p>Aplikasi web belum dibina. Jalankan <code>npm run build</code> dalam studio/web.</p>")

    return app
