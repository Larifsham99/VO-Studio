"""Settings from environment variables (all optional)."""

from __future__ import annotations

import json
import os
import secrets
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]


@dataclass
class Settings:
    data_dir: Path = field(default_factory=lambda: Path(os.environ.get("LATIF_DATA", REPO / "studio" / "data")))
    manifest: Path = field(default_factory=lambda: Path(
        os.environ.get("LATIF_MANIFEST", REPO / "benchmark" / "refs" / "manifest.yaml")))
    default_model: str = field(default_factory=lambda: os.environ.get("LATIF_MODEL", "voxcpm2"))
    device: str | None = field(default_factory=lambda: os.environ.get("LATIF_DEVICE") or None)
    # JSON object of per-adapter args, e.g. {"f5_malaysian": {"ckpt_file": "...", "vocab_file": "..."}}
    adapter_args: dict = field(default_factory=lambda: json.loads(os.environ.get("LATIF_ADAPTER_ARGS", "{}")))
    host: str = field(default_factory=lambda: os.environ.get("LATIF_HOST", "0.0.0.0"))
    port: int = field(default_factory=lambda: int(os.environ.get("LATIF_PORT", "8765")))
    web_dist: Path = field(default_factory=lambda: Path(
        os.environ.get("LATIF_WEB_DIST", REPO / "studio" / "web" / "dist")))
    max_upload_mb: int = 50
    chunk_chars: int = 300

    def __post_init__(self) -> None:
        self.data_dir = Path(self.data_dir)
        for sub in ("refs", "audio"):
            (self.data_dir / sub).mkdir(parents=True, exist_ok=True)

    @property
    def token(self) -> str:
        """Access token. Taken from LATIF_TOKEN, otherwise generated once and kept in data/token.txt."""
        env = os.environ.get("LATIF_TOKEN")
        if env:
            return env
        p = self.data_dir / "token.txt"
        if not p.exists():
            p.write_text(secrets.token_urlsafe(24))
            try:
                p.chmod(0o600)
            except OSError:
                pass
        return p.read_text().strip()
