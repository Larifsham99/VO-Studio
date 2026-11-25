"""Voice manifest + suite loading with validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


class ManifestError(ValueError):
    pass


@dataclass
class Clip:
    path: str
    transcript: str | None = None


@dataclass
class Voice:
    id: str
    permanent: bool
    reference: Clip
    holdout: str | None = None
    emotion_bank: dict[str, Clip] = field(default_factory=dict)


@dataclass
class StyleRef:
    id: str
    clip: Clip
    language: str = "ms"


@dataclass
class Manifest:
    voices: list[Voice]
    style_refs: list[StyleRef]

    @property
    def permanent(self) -> list[Voice]:
        return [v for v in self.voices if v.permanent]

    def voice(self, vid: str) -> Voice:
        for v in self.voices:
            if v.id == vid:
                return v
        raise KeyError(vid)


def _resolve(base: Path, p: str | None, what: str, required: bool = True) -> str | None:
    if not p:
        if required:
            raise ManifestError(f"{what}: missing path")
        return None
    full = (base / p).resolve()
    if not full.is_file():
        raise ManifestError(f"{what}: file not found: {full}")
    return str(full)


def _consent(entry: dict, what: str) -> None:
    c = str(entry.get("consent") or "").strip()
    if not c or c == "...":
        raise ManifestError(f"{what}: no `consent` recorded - refusing to use this recording")


def load_manifest(path: str | Path) -> Manifest:
    path = Path(path)
    base = path.parent
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    voices = []
    for v in data.get("voices", []):
        what = f"voice '{v.get('id')}'"
        _consent(v, what)
        bank = {}
        for tag, clip in (v.get("emotion_bank") or {}).items():
            bank[tag] = Clip(_resolve(base, clip.get("path"), f"{what} bank[{tag}]"), clip.get("transcript"))
        voices.append(Voice(
            id=v["id"],
            permanent=bool(v.get("permanent", False)),
            reference=Clip(_resolve(base, v.get("reference"), what), v.get("transcript")),
            holdout=_resolve(base, v.get("holdout"), f"{what} holdout", required=False),
            emotion_bank=bank,
        ))
    styles = []
    for s in data.get("style_refs", []) or []:
        what = f"style_ref '{s.get('id')}'"
        _consent(s, what)
        styles.append(StyleRef(s["id"], Clip(_resolve(base, s.get("path"), what), s.get("transcript")),
                               s.get("language", "ms")))
    if not voices:
        raise ManifestError("manifest has no voices")
    ids = [v.id for v in voices] + [s.id for s in styles]
    if len(ids) != len(set(ids)):
        raise ManifestError("duplicate ids in manifest")
    return Manifest(voices, styles)


def load_suite(path: str | Path) -> dict:
    path = Path(path)
    suite = yaml.safe_load(path.read_text(encoding="utf-8"))
    for item in suite.get("items", []):
        if "text_file" in item:
            item["text"] = (path.parent / item["text_file"]).read_text(encoding="utf-8")
    suite["_path"] = str(path)
    return suite
