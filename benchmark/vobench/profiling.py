"""Resource measurement: wall time, peak RAM (RSS incl. children), peak VRAM.

VRAM is measured two ways because they answer different questions:
* torch peak allocated  - what the model's tensors need (lower bound)
* NVML device used      - what the whole process really occupies on the card
                          (includes CUDA context, cuDNN workspaces, fragmentation)
Plan GPU purchases from the NVML number.
"""

from __future__ import annotations

import os
import platform
import threading
import time
from dataclasses import asdict, dataclass

import psutil


@dataclass
class Usage:
    wall_s: float = 0.0
    peak_rss_mb: float = 0.0
    peak_vram_torch_mb: float | None = None
    peak_vram_nvml_mb: float | None = None
    baseline_vram_nvml_mb: float | None = None

    def as_dict(self) -> dict:
        return asdict(self)


def _nvml_handle():
    try:
        import pynvml

        pynvml.nvmlInit()
        idx = int(os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0] or 0)
        return pynvml, pynvml.nvmlDeviceGetHandleByIndex(idx)
    except Exception:
        return None, None


def _torch_cuda():
    try:
        import torch

        return torch if torch.cuda.is_available() else None
    except Exception:
        return None


class ResourceMonitor:
    """Context manager sampling RSS and NVML memory every `interval` seconds."""

    def __init__(self, interval: float = 0.05):
        self.interval = interval
        self.usage = Usage()
        self._stop = threading.Event()
        self._proc = psutil.Process()
        self._nvml, self._handle = _nvml_handle()

    def _rss_mb(self) -> float:
        rss = self._proc.memory_info().rss
        for c in self._proc.children(recursive=True):
            try:
                rss += c.memory_info().rss
            except psutil.Error:
                pass
        return rss / 2**20

    def _nvml_mb(self) -> float | None:
        if not self._handle:
            return None
        try:
            return self._nvml.nvmlDeviceGetMemoryInfo(self._handle).used / 2**20
        except Exception:
            return None

    def _loop(self) -> None:
        while not self._stop.is_set():
            self.usage.peak_rss_mb = max(self.usage.peak_rss_mb, self._rss_mb())
            v = self._nvml_mb()
            if v is not None:
                self.usage.peak_vram_nvml_mb = max(self.usage.peak_vram_nvml_mb or 0.0, v)
            self._stop.wait(self.interval)

    def __enter__(self) -> "ResourceMonitor":
        torch = _torch_cuda()
        if torch:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        self.usage.baseline_vram_nvml_mb = self._nvml_mb()
        self._t0 = time.perf_counter()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        torch = _torch_cuda()
        if torch:
            torch.cuda.synchronize()
            self.usage.peak_vram_torch_mb = torch.cuda.max_memory_allocated() / 2**20
        self.usage.wall_s = time.perf_counter() - self._t0
        self._stop.set()
        self._thread.join()
        self.usage.peak_rss_mb = max(self.usage.peak_rss_mb, self._rss_mb())


def system_info() -> dict:
    info = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu": platform.processor() or platform.machine(),
        "cpu_count": psutil.cpu_count(logical=True),
        "ram_total_gb": round(psutil.virtual_memory().total / 2**30, 1),
        "gpus": [],
    }
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda"] = torch.version.cuda
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                p = torch.cuda.get_device_properties(i)
                info["gpus"].append({"name": p.name, "vram_gb": round(p.total_memory / 2**30, 1),
                                     "compute_capability": f"{p.major}.{p.minor}"})
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            info["gpus"].append({"name": "Apple MPS", "vram_gb": None})
    except ImportError:
        info["torch"] = None
    nv, h = _nvml_handle()
    if h:
        try:
            info["nvidia_driver"] = nv.nvmlSystemGetDriverVersion()
        except Exception:
            pass
    return info
