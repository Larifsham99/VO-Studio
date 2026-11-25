"""vobench command line.

  vobench sysinfo
  vobench analyze REF.wav [--transcript "..."]          style profile JSON
  vobench run --tts voxcpm2 --manifest refs/manifest.yaml [--suites malay,emotion,...]
  vobench vc  --vc seed_vc2 --manifest refs/manifest.yaml --tts-results results/voxcpm2
  vobench anchors --suite suites/malay.yaml --dir anchors/malay   (human recordings)
  vobench score   [--manifest refs/manifest.yaml]               (eval venv)
  vobench report  [--gates gates.yaml]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ALL_SUITES = ["malay", "emotion", "cloning", "style_transfer", "longform"]
HERE = Path(__file__).resolve().parent.parent


def _kv(items: list[str]) -> dict:
    out = {}
    for it in items or []:
        k, _, v = it.partition("=")
        out[k] = v
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="vobench")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("sysinfo")

    a = sub.add_parser("analyze")
    a.add_argument("wav")
    a.add_argument("--transcript")
    a.add_argument("--contour", action="store_true")

    r = sub.add_parser("run")
    r.add_argument("--tts", required=True)
    r.add_argument("--manifest", default=str(HERE / "refs" / "manifest.yaml"))
    r.add_argument("--suites", default=",".join(ALL_SUITES))
    r.add_argument("--suite-dir", default=str(HERE / "suites"))
    r.add_argument("--out", default=str(HERE / "results"))
    r.add_argument("--device")
    r.add_argument("--adapter-arg", action="append", default=[])
    r.add_argument("--routes", default="native,instruction,tts_then_vc")
    r.add_argument("--limit", type=int, help="max jobs per suite (smoke test)")
    r.add_argument("--no-resume", action="store_true")

    v = sub.add_parser("vc")
    v.add_argument("--vc", required=True)
    v.add_argument("--manifest", default=str(HERE / "refs" / "manifest.yaml"))
    v.add_argument("--tts-results", action="append", default=[])
    v.add_argument("--out", default=str(HERE / "results"))
    v.add_argument("--device")
    v.add_argument("--no-s2s", action="store_true")

    an = sub.add_parser("anchors", help="register human recordings of suite items as system 'human'")
    an.add_argument("--suite", required=True)
    an.add_argument("--dir", required=True, help="folder with <item_id>.wav files")
    an.add_argument("--voice", default="human")
    an.add_argument("--out", default=str(HERE / "results"))

    s = sub.add_parser("score")
    s.add_argument("--results", default=str(HERE / "results"))
    s.add_argument("--manifest")

    rp = sub.add_parser("report")
    rp.add_argument("--results", default=str(HERE / "results"))
    rp.add_argument("--gates", default=str(HERE / "gates.yaml"))

    args = ap.parse_args(argv)

    if args.cmd == "sysinfo":
        from .profiling import system_info

        print(json.dumps(system_info(), indent=2))
    elif args.cmd == "analyze":
        from . import audio
        from .analysis.style import analyze, to_instruction

        y, sr = audio.load(args.wav, sr=16000)
        p = analyze(y, sr, transcript=args.transcript)
        print(p.to_json(with_contour=args.contour))
        print("instruction:", to_instruction(p) or "(neutral)", file=sys.stderr)
    elif args.cmd == "run":
        from .manifest import load_manifest, load_suite
        from .run import run

        manifest = load_manifest(args.manifest)
        suites = [load_suite(Path(args.suite_dir) / f"{n}.yaml") for n in args.suites.split(",") if n]
        out = run(args.tts, suites, manifest, Path(args.out), args.device, _kv(args.adapter_arg),
                  set(args.routes.split(",")), args.limit, resume=not args.no_resume)
        print(f"done -> {out}")
    elif args.cmd == "vc":
        from .manifest import load_manifest
        from .vcpass import run_vc

        run_vc(args.vc, load_manifest(args.manifest), Path(args.out), [Path(p) for p in args.tts_results],
               args.device, do_s2s=not args.no_s2s)
    elif args.cmd == "anchors":
        from . import audio
        from .manifest import load_suite

        suite = load_suite(args.suite)
        d = Path(args.out) / "human"
        d.mkdir(parents=True, exist_ok=True)
        n = 0
        with (d / "records.jsonl").open("a", encoding="utf-8") as f:
            for it in suite["items"]:
                src = Path(args.dir) / f"{it['id']}.wav"
                if not src.exists():
                    continue
                y, sr = audio.load(src)
                dst = d / suite["name"] / f"{it['id']}__{args.voice}__human.wav"
                audio.save_wav(dst, y, sr)
                f.write(json.dumps({"key": f"{suite['name']}/{it['id']}__{args.voice}__human", "suite": suite["name"],
                                    "item": it["id"], "voice": args.voice, "variant": "human", "adapter": "human",
                                    "language": it.get("language", suite.get("language", "ms")),
                                    "text": it["text"], "wav": str(dst.relative_to(d)), "status": "ok",
                                    "audio_s": audio.duration(y, sr), "refs": {}}, ensure_ascii=False) + "\n")
                n += 1
        print(f"registered {n} human anchor recordings")
    elif args.cmd == "score":
        from .score import score_all, voice_baselines

        root = Path(args.results)
        if args.manifest:
            from .manifest import load_manifest

            (root / "voice_baselines.json").write_text(json.dumps(voice_baselines(load_manifest(args.manifest)), indent=2))
        print(f"scores -> {score_all(root)}")
    elif args.cmd == "report":
        from .report import build_report

        print(f"report -> {build_report(Path(args.results), Path(args.gates))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
