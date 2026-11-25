"""Score generated audio. Run in the eval venv:  vobench score results/"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from . import audio
from .analysis.style import analyze, style_distance
from .metrics import backends as B
from .metrics.prosody import contour_correlation, delivery_checks
from .text.emotion_tags import EMOTIONS, parse
from .text.malay_normalize import normalize_for_wer

TAG_WORDS = set(EMOTIONS) | {"whisper", "shout", "laugh", "sigh", "bracket"}


def latest_records(d: Path) -> dict[str, dict]:
    recs: dict[str, dict] = {}
    p = d / "records.jsonl"
    if not p.exists():
        return recs
    for line in p.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        prev = recs.get(r["key"])
        if prev is None or r.get("status") == "ok" or prev.get("status") != "ok":
            recs[r["key"]] = r
    return recs


def _wer_cer(ref: str, hyp: str) -> tuple[float, float]:
    import jiwer

    r, h = normalize_for_wer(ref), normalize_for_wer(hyp)
    if not r:
        return float("nan"), float("nan")
    return float(jiwer.wer(r, h)), float(jiwer.cer(r, h))


_profile_cache: dict[str, object] = {}


def profile(path: str, transcript: str | None = None):
    k = f"{path}|{transcript}"
    if k not in _profile_cache:
        y, sr = audio.load(path, sr=16000)
        _profile_cache[k] = analyze(y, sr, transcript=normalize_for_wer(transcript) if transcript else None)
    return _profile_cache[k]


def score_record(d: Path, r: dict, all_recs: dict[str, dict]) -> dict:
    s: dict = {"key": r["key"], "adapter": r["adapter"], "suite": r["suite"], "item": r["item"],
               "voice": r["voice"], "variant": r["variant"], "language": r["language"],
               "rtf": r.get("rtf_total") or r.get("rtf"), "audio_s": r.get("audio_s"),
               "peak_vram_mb": (r.get("usage") or {}).get("peak_vram_nvml_mb"),
               "peak_rss_mb": (r.get("usage") or {}).get("peak_rss_mb")}
    wav = str(d / r["wav"])
    refs = r.get("refs", {})
    text = r.get("text") or ""
    s["text"] = text

    hyp = B.transcribe(wav, r["language"])
    if hyp is not None:
        s["asr"] = hyp
        s["wer"], s["cer"] = _wer_cer(text, hyp)
        hyp_words = set(re.findall(r"\w+", hyp.lower()))
        text_words = set(re.findall(r"\w+", text.lower()))
        s["tag_read_aloud"] = bool((hyp_words - text_words) & TAG_WORDS)
    s["mos"] = B.mos(wav)
    s["sim_target"] = B.speaker_sim(wav, refs.get("target_ref"))
    s["sim_holdout"] = B.speaker_sim(wav, refs.get("holdout_ref"))

    if r["suite"] == "emotion":
        tagged = r.get("meta", {}).get("tagged_text", "")
        segs = parse(tagged).segments
        main_tag = next((t for sg in segs for t in sg.tags), None)
        s["tag"] = main_tag
        emo = B.emotion(wav)
        if emo:
            s["ser_label"], s["ser_score"] = emo["label"], emo["score"]
            targets = EMOTIONS[main_tag].ser_targets if main_tag in EMOTIONS else frozenset()
            s["ser_hit"] = (emo["label"] in targets) if targets else None
        if r["variant"] != "neutral_baseline" and main_tag:
            base_key = r["key"].rsplit("__", 1)[0] + "__neutral_baseline"
            base = all_recs.get(base_key)
            if base and base.get("status") == "ok":
                s.update(delivery_checks(main_tag, profile(wav, text), profile(str(d / base["wav"]), base.get("text"))))

    if r["suite"] == "style_transfer" and refs.get("style_ref"):
        style = refs["style_ref"]
        s["sim_style_speaker"] = B.speaker_sim(wav, style)  # identity leakage: lower is better
        pg, ps = profile(wav, text), profile(style, text if refs.get("same_text") else None)
        s["style_distance"] = style_distance(pg, ps)["mean"]
        if refs.get("same_text"):
            yg, _ = audio.load(wav, sr=16000)
            ys, _ = audio.load(style, sr=16000)
            s["f0_contour_corr"] = contour_correlation(yg, ys, 16000, pg, ps)

    if r.get("chunks"):
        first = str(d / r["chunks"][0]["wav"])
        sims, cers = [], []
        for c in r["chunks"]:
            cw = str(d / c["wav"])
            sims.append(B.speaker_sim(cw, first))
            h = B.transcribe(cw, r["language"])
            if h is not None:
                cers.append(_wer_cer(c["text"], h)[1])
        sims_v = [x for x in sims[1:] if x is not None]
        s["chunk_count"] = len(r["chunks"])
        s["drift_min_sim_to_first"] = min(sims_v) if sims_v else None
        s["chunk_cer_max"] = max(cers) if cers else None
        s["chunk_cer_mean"] = float(np.mean(cers)) if cers else None
    return s


def voice_baselines(manifest) -> dict:
    """Human ceiling: similarity between two REAL clips of the same speaker,
    plus how similar the style-reference speakers are to each permanent voice
    (if they are already similar, a low leakage score proves nothing)."""
    out = {"same_speaker": {}, "style_vs_voice": {}}
    for v in manifest.voices:
        out["same_speaker"][v.id] = B.speaker_sim(v.reference.path, v.holdout)
        for s in manifest.style_refs:
            out["style_vs_voice"][f"{s.id}|{v.id}"] = B.speaker_sim(s.clip.path, v.reference.path)
    return out


def score_all(results_root: Path) -> Path:
    out = results_root / "scores.jsonl"
    existing = {}
    if out.exists():
        for line in out.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            existing[row["key"] + "|" + row["adapter"]] = row
    rows = []
    for d in sorted(p for p in results_root.iterdir() if (p / "records.jsonl").exists()):
        recs = latest_records(d)
        for r in recs.values():
            if r.get("status") != "ok":
                rows.append({"key": r["key"], "adapter": r.get("adapter", d.name), "suite": r.get("suite"),
                             "status": r.get("status"), "error": r.get("error"), "items": r.get("items")})
                continue
            k = r["key"] + "|" + r["adapter"]
            if k in existing and existing[k].get("status", "ok") == "ok":
                rows.append(existing[k])
                continue
            print(f"[score] {d.name}: {r['key']}")
            row = score_record(d, r, recs)
            row["status"] = "ok"
            rows.append(row)
    with out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, default=float) + "\n")
    return out
