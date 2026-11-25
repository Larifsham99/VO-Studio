"""Voice-conversion pass, run in the VC model's own venv.

1. Finishes `tts_then_vc` jobs left by a TTS run (status == needs_vc).
2. Runs the `s2s` Mode B route: converts each style reference recording itself
   into each permanent voice (same words, reference prosody, new identity).
"""

from __future__ import annotations

import json
from pathlib import Path

from . import audio
from .adapters import get_vc
from .adapters.base import pick_device
from .manifest import Manifest
from .profiling import ResourceMonitor


def _convert(vc, src: str, target: str, out_path: Path, kw: dict) -> dict:
    with ResourceMonitor() as mon:
        wav, sr = vc.convert(src, target, **kw)
    audio.save_wav(out_path, wav, sr)
    dur = audio.duration(wav, sr)
    return {"sr": sr, "audio_s": dur, "usage": mon.usage.as_dict(), "rtf": mon.usage.wall_s / dur if dur else None}


def run_vc(vc_name: str, manifest: Manifest, out_root: Path, tts_dirs: list[Path], device: str | None,
           do_s2s: bool = True, vc_kw: dict | None = None) -> None:
    device = pick_device(device)
    vc = get_vc(vc_name)
    vc.load(device)
    vc_kw = vc_kw or {}

    for d in tts_dirs:
        rec_path = d / "records.jsonl"
        recs = [json.loads(line) for line in rec_path.read_text().splitlines()]
        done = {r["key"] for r in recs if r.get("status") == "ok"}
        new = []
        for r in recs:
            if r.get("status") != "needs_vc":
                continue
            key = r["key"].replace("__tts_then_vc", f"__tts_then_vc+{vc.name}")
            if key in done:
                continue
            out = d / f"{key}.wav"
            nr = {**r, "key": key, "variant": f"tts_then_vc+{vc.name}", "intermediate_wav": r["wav"],
                  "wav": str(out.relative_to(d)), "vc": vc.name, "vc_license": vc.license}
            try:
                info = _convert(vc, str(d / r["wav"]), r["vc_target_ref"], out, vc_kw)
                nr.update(info, status="ok", tts_usage=r.get("usage"),
                          rtf_total=((r["usage"]["wall_s"] + info["usage"]["wall_s"]) / info["audio_s"])
                          if info["audio_s"] else None)
            except Exception as e:
                nr.update(status="error", error=f"{type(e).__name__}: {e}")
            new.append(nr)
            print(f"[vc] {key}: {nr['status']}")
        with rec_path.open("a", encoding="utf-8") as f:
            for nr in new:
                f.write(json.dumps(nr, ensure_ascii=False) + "\n")

    if not do_s2s:
        return
    d = out_root / vc.name
    d.mkdir(parents=True, exist_ok=True)
    with (d / "records.jsonl").open("a", encoding="utf-8") as f:
        for v in manifest.permanent:
            for s in manifest.style_refs:
                key = f"style_transfer/st_same_text__{s.id}__{v.id}__s2s"
                out = d / f"{key}.wav"
                rec = {"key": key, "suite": "style_transfer", "item": f"st_same_text__{s.id}", "voice": v.id,
                       "variant": "s2s", "language": s.language, "adapter": vc.name, "license": vc.license,
                       "text": s.clip.transcript or "", "wav": str(out.relative_to(d)),
                       "refs": {"target_ref": v.reference.path, "holdout_ref": v.holdout,
                                "style_ref": s.clip.path, "same_text": True}}
                try:
                    rec.update(_convert(vc, s.clip.path, v.reference.path, out, vc_kw), status="ok")
                except Exception as e:
                    rec.update(status="error", error=f"{type(e).__name__}: {e}")
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                print(f"[s2s] {key}: {rec['status']}")
