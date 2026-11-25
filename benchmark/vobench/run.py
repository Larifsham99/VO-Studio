"""Generation runner: suites x voices x strategies -> wav files + records.jsonl.

Run inside the venv of ONE TTS model. Every job's outcome is recorded,
including failures; nothing is skipped silently.
"""

from __future__ import annotations

import json
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path

from . import audio
from .adapters import get_tts
from .adapters.base import SynthesisRequest, TTSAdapter, pick_device
from .manifest import Manifest, Voice
from .profiling import ResourceMonitor, system_info
from .text import malay_normalize
from .text.emotion_tags import EMOTIONS, parse
from .text.segment import chunk_text


@dataclass
class Job:
    suite: str
    item_id: str
    voice_id: str
    variant: str  # strategy / route / normalisation label
    text: str  # tagged or plain input text
    language: str
    request: dict  # SynthesisRequest kwargs except text/tags
    refs: dict = field(default_factory=dict)  # scoring references
    meta: dict = field(default_factory=dict)
    chunk_max_chars: int | None = None  # per-segment generation when set
    needs_vc: str | None = None  # path of target voice ref for a later VC pass

    @property
    def key(self) -> str:
        return f"{self.suite}/{self.item_id}__{self.voice_id}__{self.variant}"


def _strategies(adapter: TTSAdapter, voice: Voice) -> list[str]:
    s = []
    if adapter.supports_instruction:
        s.append("instruction")
    if adapter.supports_exaggeration:
        s.append("exaggeration")
    if voice.emotion_bank:
        s.append("bank")
    return s


def _voice_req(v: Voice) -> dict:
    return {"speaker_ref": v.reference.path, "speaker_ref_text": v.reference.transcript}


def _voice_refs(v: Voice) -> dict:
    return {"target_ref": v.reference.path, "holdout_ref": v.holdout}


def build_jobs(suite: dict, manifest: Manifest, adapter: TTSAdapter, routes: set[str]) -> list[Job]:
    jobs: list[Job] = []
    kind, name = suite["type"], suite["name"]
    default_lang = suite.get("language", "ms")
    permanent = manifest.permanent or manifest.voices[:1]

    for item in suite["items"]:
        lang = item.get("language", default_lang)
        if adapter.languages and lang not in adapter.languages and kind != "style_transfer":
            continue  # recorded in the report as "language unsupported"

        if kind == "tts":
            for v in permanent:
                variants = {"raw": item["text"]}
                if suite.get("normalize") in ("both", True) and lang == "ms":
                    norm = malay_normalize.normalize(item["text"])
                    if norm != item["text"]:
                        variants["normalized"] = norm
                for var, text in variants.items():
                    jobs.append(Job(name, item["id"], v.id, var, text, lang,
                                    {**_voice_req(v), "emotion_strategy": "none"},
                                    _voice_refs(v), {"focus": item.get("focus"), "ref_text": item["text"]}))

        elif kind == "emotion":
            for v in permanent:
                plain = parse(item["text"]).plain_text
                jobs.append(Job(name, item["id"], v.id, "neutral_baseline", plain, lang,
                                {**_voice_req(v), "emotion_strategy": "none"}, _voice_refs(v),
                                {"tagged_text": item["text"]}))
                for strat in _strategies(adapter, v):
                    jobs.append(Job(name, item["id"], v.id, strat, item["text"], lang,
                                    {**_voice_req(v), "emotion_strategy": strat}, _voice_refs(v),
                                    {"tagged_text": item["text"]}, chunk_max_chars=400))

        elif kind == "clone":
            modes = ["isolated", "ultimate"] if adapter.supports_transcript_prompt and adapter.name == "voxcpm2" else ["default"]
            for v in manifest.voices:
                for mode in modes:
                    jobs.append(Job(name, item["id"], v.id, mode, item["text"], lang,
                                    {**_voice_req(v), "emotion_strategy": "none",
                                     "params": {"clone_mode": mode}},
                                    _voice_refs(v), {"permanent": v.permanent}))

        elif kind == "style_transfer":
            for v in permanent:
                for s in manifest.style_refs:
                    text = s.clip.transcript if item.get("same_text") else item.get("text")
                    lang_i = s.language if item.get("same_text") else lang
                    if not text or (adapter.languages and lang_i not in adapter.languages):
                        continue
                    base_refs = {**_voice_refs(v), "style_ref": s.clip.path,
                                 "same_text": bool(item.get("same_text"))}
                    sid = f"{item['id']}__{s.id}"
                    if "native" in routes and adapter.supports_style_ref and s.clip.transcript:
                        jobs.append(Job(name, sid, v.id, "native", text, lang_i,
                                        {**_voice_req(v), "style_ref": s.clip.path,
                                         "style_ref_text": s.clip.transcript, "emotion_strategy": "none",
                                         "params": {"clone_mode": "style_cont"}}, base_refs))
                    if "instruction" in routes and adapter.supports_instruction:
                        jobs.append(Job(name, sid, v.id, "instruction", text, lang_i,
                                        {**_voice_req(v), "emotion_strategy": "none",
                                         "params": {"instruction_from_style_ref": s.clip.path,
                                                    "style_ref_text": s.clip.transcript}}, base_refs))
                    if "tts_then_vc" in routes:
                        mode = "ultimate" if adapter.name == "voxcpm2" else "default"
                        jobs.append(Job(name, sid, v.id, "tts_then_vc", text, lang_i,
                                        {"speaker_ref": s.clip.path, "speaker_ref_text": s.clip.transcript,
                                         "emotion_strategy": "none", "params": {"clone_mode": mode}},
                                        base_refs, needs_vc=v.reference.path))

        elif kind == "longform":
            for v in permanent:
                strats = _strategies(adapter, v)
                strat = strats[0] if strats else "none"
                jobs.append(Job(name, item["id"], v.id, strat, item["text"], lang,
                                {**_voice_req(v), "emotion_strategy": strat}, _voice_refs(v),
                                {"longform": True}, chunk_max_chars=suite.get("max_chars", 280)))
    return jobs


def _bank_ref(voice: Voice | None, tags: list[str]):
    if not voice:
        return None
    for t in tags:  # event tags last in Segment.tags, so state emotion wins
        if t in voice.emotion_bank:
            return voice.emotion_bank[t]
    return None


def _synth_text(adapter: TTSAdapter, job: Job, voice: Voice | None, text: str, tags: list[str]):
    kw = dict(job.request)
    params = dict(kw.pop("params", {}) or {})
    if "instruction_from_style_ref" in params:
        from .analysis.style import analyze, to_instruction

        y, sr = audio.load(params.pop("instruction_from_style_ref"), sr=16000)
        prof = analyze(y, sr, transcript=params.pop("style_ref_text", None))
        params["instruction"] = to_instruction(prof)
        job.meta["derived_instruction"] = params["instruction"]
    if kw.get("emotion_strategy") == "bank":
        clip = _bank_ref(voice, tags)
        if clip:  # emotion clip of the SAME voice becomes the reference
            kw["speaker_ref"], kw["speaker_ref_text"] = clip.path, clip.transcript
    req = SynthesisRequest(text=text, language=job.language, tags=tags, params=params, **kw)
    return adapter.synthesize(req)


def execute(adapter: TTSAdapter, job: Job, manifest: Manifest, out_dir: Path) -> dict:
    voice = next((v for v in manifest.voices if v.id == job.voice_id), None)
    wav_path = out_dir / f"{job.key}.wav"
    rec = {"key": job.key, "suite": job.suite, "item": job.item_id, "voice": job.voice_id,
           "variant": job.variant, "language": job.language, "adapter": adapter.name,
           "input_text": job.text, "refs": job.refs, "meta": job.meta}
    try:
        with ResourceMonitor() as mon:
            if job.chunk_max_chars:
                chunks = chunk_text(job.text, job.chunk_max_chars)
                waves, sr, chunk_info = [], None, []
                for i, ch in enumerate(chunks):
                    if not ch.text:
                        continue
                    t0 = time.perf_counter()
                    w, sr = _synth_text(adapter, job, voice, ch.text, ch.tags)
                    waves.append(w)
                    cpath = out_dir / f"{job.key}.chunks" / f"{i:03d}.wav"
                    audio.save_wav(cpath, w, sr)
                    chunk_info.append({"i": i, "text": ch.text, "tags": ch.tags,
                                       "wav": str(cpath.relative_to(out_dir)),
                                       "audio_s": audio.duration(w, sr),
                                       "wall_s": time.perf_counter() - t0})
                wav = audio.concat(waves, sr)
                rec["chunks"] = chunk_info
                spoken = " ".join(c["text"] for c in chunk_info)
            else:
                pr = parse(job.text)
                tags = pr.segments[0].tags if pr.segments else []
                spoken = pr.plain_text
                wav, sr = _synth_text(adapter, job, voice, spoken, tags)
                if pr.unknown_tags:
                    rec["unknown_tags"] = pr.unknown_tags
        audio.save_wav(wav_path, wav, sr)
        dur = audio.duration(wav, sr)
        rec.update(status="needs_vc" if job.needs_vc else "ok", wav=str(wav_path.relative_to(out_dir)),
                   sr=sr, audio_s=dur, text=spoken, usage=mon.usage.as_dict(),
                   rtf=(mon.usage.wall_s / dur) if dur else None)
        if job.needs_vc:
            rec["vc_target_ref"] = job.needs_vc
    except Exception as e:  # recorded, never hidden
        rec.update(status="error", error=f"{type(e).__name__}: {e}", trace=traceback.format_exc()[-2000:])
    return rec


def run(tts: str, suites: list[dict], manifest: Manifest, out_root: Path, device: str | None,
        adapter_args: dict, routes: set[str], limit: int | None = None, resume: bool = True) -> Path:
    device = pick_device(device)
    adapter = get_tts(tts, **adapter_args)
    out_dir = out_root / adapter.name
    out_dir.mkdir(parents=True, exist_ok=True)
    rec_path = out_dir / "records.jsonl"
    done = set()
    if resume and rec_path.exists():
        for line in rec_path.read_text().splitlines():
            r = json.loads(line)
            if r.get("status") in ("ok", "needs_vc"):
                done.add(r["key"])

    with ResourceMonitor() as load_mon:
        adapter.load(device)
    meta = {"adapter": adapter.name, "license": adapter.license, "device": device,
            "system": system_info(), "load": load_mon.usage.as_dict(), "adapter_args": adapter_args,
            "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    (out_dir / f"run_{time.strftime('%Y%m%d_%H%M%S')}.json").write_text(json.dumps(meta, indent=2))
    print(f"[vobench] {adapter.name} loaded on {device} in {load_mon.usage.wall_s:.1f}s "
          f"(peak VRAM nvml={load_mon.usage.peak_vram_nvml_mb} MB)")

    with rec_path.open("a", encoding="utf-8") as f:
        for suite in suites:
            jobs = build_jobs(suite, manifest, adapter, routes)
            if limit:
                jobs = jobs[:limit]
            skipped_lang = [i["id"] for i in suite["items"]
                            if adapter.languages and i.get("language", suite.get("language", "ms"))
                            not in adapter.languages and suite["type"] != "style_transfer"]
            if skipped_lang:
                f.write(json.dumps({"key": f"{suite['name']}/_unsupported", "suite": suite["name"],
                                    "status": "unsupported_language", "items": skipped_lang,
                                    "adapter": adapter.name}) + "\n")
            for n, job in enumerate(jobs, 1):
                if job.key in done:
                    continue
                rec = execute(adapter, job, manifest, out_dir)
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                rtf = rec.get("rtf")
                print(f"[{suite['name']} {n}/{len(jobs)}] {job.key}: {rec['status']}"
                      + (f" rtf={rtf:.2f}" if rtf else "") + (f" {rec.get('error')}" if rec.get("error") else ""))
    return out_dir


def known_emotions() -> list[str]:
    return sorted(EMOTIONS)
