"""Adapter registry. Imports are lazy so each model's venv only needs its own deps."""

from __future__ import annotations

import importlib

TTS = {
    "chatterbox_ml": ("vobench.adapters.chatterbox_ml", "ChatterboxMultilingual"),
    "voxcpm2": ("vobench.adapters.voxcpm2", "VoxCPM2"),
    "f5_malaysian": ("vobench.adapters.f5_malaysian", "F5Malaysian"),
}

VC = {
    "seed_vc2": ("vobench.adapters.vc", "SeedVC2"),
    "chatterbox_vc": ("vobench.adapters.vc", "ChatterboxVC"),
}


def _load(table: dict, name: str, **kwargs):
    if name not in table:
        raise KeyError(f"unknown adapter '{name}'. Known: {', '.join(sorted(table))}")
    mod, cls = table[name]
    return getattr(importlib.import_module(mod), cls)(**kwargs)


def get_tts(name: str, **kwargs):
    return _load(TTS, name, **kwargs)


def get_vc(name: str, **kwargs):
    return _load(VC, name, **kwargs)
