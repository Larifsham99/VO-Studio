"""Aggregate scores.jsonl -> REPORT.md (+ gate verdicts) and listening.html."""

from __future__ import annotations

import html
import json
import math
import random
from collections import defaultdict
from pathlib import Path

import yaml


def _mean(xs):
    xs = [x for x in xs if isinstance(x, (int, float)) and not (isinstance(x, float) and math.isnan(x))]
    return sum(xs) / len(xs) if xs else None


def _rate(xs):
    xs = [x for x in xs if x is not None]
    return sum(1 for x in xs if x) / len(xs) if xs else None


def _max(xs):
    xs = [x for x in xs if isinstance(x, (int, float))]
    return max(xs) if xs else None


def _f(x, pct=False, nd=2):
    if x is None:
        return "–"
    return f"{100 * x:.0f}%" if pct else f"{x:.{nd}f}"


def load_scores(root: Path) -> list[dict]:
    return [json.loads(line) for line in (root / "scores.jsonl").read_text(encoding="utf-8").splitlines()]


def _group(rows, *keys):
    g = defaultdict(list)
    for r in rows:
        g[tuple(r.get(k) for k in keys)].append(r)
    return g


def _table(header: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def build_report(root: Path, gates_path: Path | None = None) -> Path:
    rows = load_scores(root)
    ok = [r for r in rows if r.get("status") == "ok"]
    bad = [r for r in rows if r.get("status") not in ("ok",)]
    md = ["# Latif Voice AI – model benchmark report", ""]

    sysinfo = {}
    for p in root.glob("*/run_*.json"):
        m = json.loads(p.read_text())
        sysinfo[m["adapter"]] = m
    if sysinfo:
        md += ["## System and load cost", ""]
        trs = []
        for a, m in sorted(sysinfo.items()):
            gpu = ", ".join(f"{g['name']} {g.get('vram_gb')} GB" for g in m["system"].get("gpus", [])) or "none"
            ld = m["load"]
            trs.append([a, m["license"], m["device"], gpu, _f(ld["wall_s"], nd=1),
                        _f(ld.get("peak_vram_nvml_mb"), nd=0), _f(ld.get("peak_rss_mb"), nd=0)])
        md += [_table(["adapter", "license", "device", "GPU", "load s", "VRAM after load MB (NVML)", "RAM MB"], trs), ""]

    def overall(suite: str, cols: list[tuple[str, callable]]) -> None:
        sub = [r for r in ok if r["suite"] == suite]
        if not sub:
            return
        md.extend([f"## {suite}", ""])
        trs = []
        for (a, v), rs in sorted(_group(sub, "adapter", "variant").items()):
            trs.append([a, v, str(len(rs))] + [fn(rs) for _, fn in cols])
        md.extend([_table(["adapter", "variant", "n"] + [c for c, _ in cols], trs), ""])

    common = [
        ("CER", lambda rs: _f(_mean(r.get("cer") for r in rs), pct=True)),
        ("WER", lambda rs: _f(_mean(r.get("wer") for r in rs), pct=True)),
        ("UTMOS", lambda rs: _f(_mean(r.get("mos") for r in rs))),
        ("sim→holdout", lambda rs: _f(_mean(r.get("sim_holdout") for r in rs))),
        ("RTF", lambda rs: _f(_mean(r.get("rtf") for r in rs))),
        ("peak VRAM MB", lambda rs: _f(_max(r.get("peak_vram_mb") for r in rs), nd=0)),
    ]
    overall("malay", common)
    sub = [r for r in ok if r["suite"] == "malay"]
    if sub:
        md += ["### Malay CER by focus area (raw text)", ""]
        foci = sorted({r["item"].rsplit("_", 1)[0] for r in sub})
        trs = []
        for (a,), rs in sorted(_group([r for r in sub if r["variant"] == "raw"], "adapter").items()):
            trs.append([a] + [_f(_mean(r.get("cer") for r in rs if r["item"].startswith(fo)), pct=True) for fo in foci])
        md += [_table(["adapter"] + foci, trs), ""]

    overall("emotion", [
        ("SER hit", lambda rs: _f(_rate(r.get("ser_hit") for r in rs), pct=True)),
        ("direction ok", lambda rs: _f(_rate(r.get("direction_ok") for r in rs), pct=True)),
        ("tag read aloud", lambda rs: str(sum(1 for r in rs if r.get("tag_read_aloud")))),
        ("CER", lambda rs: _f(_mean(r.get("cer") for r in rs), pct=True)),
        ("UTMOS", lambda rs: _f(_mean(r.get("mos") for r in rs))),
        ("sim→holdout", lambda rs: _f(_mean(r.get("sim_holdout") for r in rs))),
    ])
    emo = [r for r in ok if r["suite"] == "emotion" and r["variant"] != "neutral_baseline" and r.get("tag")]
    if emo:
        md += ["### Emotion detail per tag (direction ok / SER label)", ""]
        tags = sorted({r["tag"] for r in emo})
        trs = []
        for (a, v), rs in sorted(_group(emo, "adapter", "variant").items()):
            cells = []
            for t in tags:
                tr = [r for r in rs if r["tag"] == t]
                d = _rate(r.get("direction_ok") for r in tr)
                lab = ",".join(sorted({r.get("ser_label") or "?" for r in tr}))
                cells.append(f"{_f(d, pct=True)} {lab}")
            trs.append([a, v] + cells)
        md += [_table(["adapter", "variant"] + tags, trs), ""]

    overall("cloning", common)
    overall("style_transfer", [
        ("sim→permanent", lambda rs: _f(_mean(r.get("sim_holdout") or r.get("sim_target") for r in rs))),
        ("sim→style spk (leak)", lambda rs: _f(_mean(r.get("sim_style_speaker") for r in rs))),
        ("style dist ↓", lambda rs: _f(_mean(r.get("style_distance") for r in rs))),
        ("F0 corr (same text)", lambda rs: _f(_mean(r.get("f0_contour_corr") for r in rs))),
        ("CER", lambda rs: _f(_mean(r.get("cer") for r in rs), pct=True)),
        ("UTMOS", lambda rs: _f(_mean(r.get("mos") for r in rs))),
    ])
    overall("longform", [
        ("chunks", lambda rs: _f(_mean(r.get("chunk_count") for r in rs), nd=0)),
        ("min sim to chunk0", lambda rs: _f(_mean(r.get("drift_min_sim_to_first") for r in rs))),
        ("chunk CER mean", lambda rs: _f(_mean(r.get("chunk_cer_mean") for r in rs), pct=True)),
        ("chunk CER max", lambda rs: _f(_mean(r.get("chunk_cer_max") for r in rs), pct=True)),
        ("RTF", lambda rs: _f(_mean(r.get("rtf") for r in rs))),
        ("peak VRAM MB", lambda rs: _f(_max(r.get("peak_vram_mb") for r in rs), nd=0)),
    ])

    base = root / "voice_baselines.json"
    if base.exists():
        b = json.loads(base.read_text())
        md += ["## Human ceilings", "", "Same-speaker similarity between two real recordings "
               "(a cloned voice cannot be expected to exceed this):", ""]
        md += [_table(["voice", "sim(reference, holdout)"],
                      [[k, _f(v)] for k, v in b["same_speaker"].items()]), ""]

    if gates_path and gates_path.exists():
        md += gate_section(ok, yaml.safe_load(gates_path.read_text()),
                           json.loads(base.read_text()) if base.exists() else None)

    if bad:
        md += ["## Failures and unsupported items", ""]
        md += [_table(["adapter", "key", "status", "detail"],
                      [[str(r.get("adapter")), r["key"], str(r.get("status")),
                        html.escape(str(r.get("error") or r.get("items") or ""))[:160]] for r in bad]), ""]

    md += ["", "_Automatic metrics screen candidates; the blind listening test (listening.html) decides._"]
    out = root / "REPORT.md"
    out.write_text("\n".join(md), encoding="utf-8")
    build_listening(root, ok)
    return out


def gate_section(ok: list[dict], gates: dict, baselines: dict | None) -> list[str]:
    md = ["## Viability gates", "", f"Thresholds from `gates.yaml`. A model is viable only if every gate passes.", ""]
    ceiling = None
    if baselines:
        vals = [v for v in baselines["same_speaker"].values() if v is not None]
        ceiling = sum(vals) / len(vals) if vals else None
    trs = []
    for (a,), rs in sorted(_group(ok, "adapter").items()):
        def suite(s, variant_ne=None):
            return [r for r in rs if r["suite"] == s and (variant_ne is None or r["variant"] != variant_ne)]

        checks = {}
        # the app will normalise text, so take each item's best front-end variant
        per_item = defaultdict(list)
        for r in suite("malay"):
            if r.get("cer") is not None:
                per_item[(r["item"], r["voice"])].append(r["cer"])
        cer = _mean(min(v) for v in per_item.values())
        checks["malay_cer"] = (cer, cer is not None and cer <= gates["malay_cer_max"])
        sim = _mean(r.get("sim_holdout") for r in suite("cloning"))
        thr = gates["clone_sim_min_ratio_of_human"] * ceiling if ceiling else gates.get("clone_sim_abs_min", 0.5)
        checks["clone_sim"] = (sim, sim is not None and sim >= thr)
        e = suite("emotion", "neutral_baseline")
        dirok = _rate(r.get("direction_ok") for r in e)
        checks["emotion_direction"] = (dirok, dirok is not None and dirok >= gates["emotion_direction_min"])
        leaks = sum(1 for r in e if r.get("tag_read_aloud"))
        checks["no_tag_read_aloud"] = (leaks, leaks <= gates.get("tag_read_aloud_max", 0))
        mos = _mean(r.get("mos") for r in rs)
        checks["utmos"] = (mos, mos is not None and mos >= gates["utmos_min"])
        rtf = _mean(r.get("rtf") for r in rs)
        checks["rtf"] = (rtf, rtf is not None and rtf <= gates["rtf_max"])
        vram = _max(r.get("peak_vram_mb") for r in rs)
        checks["vram"] = (vram, vram is None or vram <= gates["vram_mb_max"])
        verdict = "PASS" if all(v[1] for v in checks.values()) else "FAIL"
        trs.append([a] + [f"{'✅' if p else '❌'} {_f(v) if isinstance(v, float) else v}" for v, p in checks.values()] + [verdict])
    md += [_table(["adapter", "Malay CER", "clone sim", "emotion dir", "tags read aloud", "UTMOS", "RTF", "peak VRAM", "verdict"], trs), ""]
    return md


def build_listening(root: Path, ok: list[dict], seed: int = 7) -> Path:
    """Blind A/B/C... rating page. Systems are shuffled per row; the key is
    hidden until 'reveal' so ratings aren't biased by model names."""
    rng = random.Random(seed)
    groups = _group([r for r in ok if r["variant"] != "neutral_baseline"], "suite", "item", "voice")
    blocks, key = [], {}
    for (suite, item, voice), rs in sorted(groups.items(), key=lambda kv: tuple(map(str, kv[0]))):
        rs = rs[:]
        rng.shuffle(rs)
        rid = f"{suite}/{item}/{voice}"
        text = html.escape(rs[0].get("text") or "")
        cells = []
        for i, r in enumerate(rs):
            lab = chr(65 + i)
            key[f"{rid}/{lab}"] = f"{r['adapter']}:{r['variant']}"
            src = html.escape(f"{r['adapter']}/{r['key']}.wav")
            cells.append(
                f'<div class="sys"><b>{lab}</b><audio controls preload="none" src="{src}"></audio>'
                + "".join(f'<label>{m}<select data-row="{html.escape(rid)}" data-sys="{lab}" data-m="{m}">'
                          f'<option></option>{"".join(f"<option>{n}</option>" for n in range(1, 6))}</select></label>'
                          for m in ("natural", "emotion", "similarity"))
                + "</div>")
        blocks.append(f'<section><h3>{html.escape(rid)}</h3><p class="t">{text}</p>{"".join(cells)}</section>')
    page = f"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Blind Listening Test</title><style>
:root{{--bg:#fafafa;--fg:#1d1d1f;--card:#fff;--line:#ddd}}
@media (prefers-color-scheme:dark){{:root{{--bg:#151515;--fg:#eee;--card:#222;--line:#333}}}}
body{{background:var(--bg);color:var(--fg);font:15px/1.4 system-ui,sans-serif;margin:0 auto;max-width:960px;padding:16px}}
section{{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:12px;margin:12px 0}}
.sys{{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin:6px 0}} audio{{max-width:100%}}
label{{font-size:13px}} .t{{opacity:.75;font-size:13px}} button{{margin:4px}}
</style></head><body>
<h1>Blind listening test</h1>
<p>Rate each system 1–5: <b>natural</b> (human-like), <b>emotion</b> (matches the intended delivery),
<b>similarity</b> (sounds like the target voice). Use headphones. Ratings stay in this page until you export.</p>
<button id="exp">Export CSV</button><button id="rev">Reveal systems</button>
{''.join(blocks)}
<script>
const KEY={json.dumps(key)};
document.getElementById('exp').onclick=()=>{{
 const rows=[['row','label','system','metric','score']];
 document.querySelectorAll('select').forEach(s=>{{if(s.value)rows.push([s.dataset.row,s.dataset.sys,KEY[s.dataset.row+'/'+s.dataset.sys],s.dataset.m,s.value])}});
 const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([rows.map(r=>r.join(',')).join('\\n')],{{type:'text/csv'}}));
 a.download='listening_ratings.csv';a.click();}};
document.getElementById('rev').onclick=()=>{{if(!confirm('Reveal model names? Finish rating first.'))return;
 document.querySelectorAll('.sys').forEach(d=>{{const s=d.querySelector('select');d.querySelector('b').textContent+=' = '+KEY[s.dataset.row+'/'+s.dataset.sys]}})}};
</script></body></html>"""
    out = root / "listening.html"
    out.write_text(page, encoding="utf-8")
    return out
