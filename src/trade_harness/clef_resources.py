"""Offline planning checks, without downloading or importing model weights.

Budgets are conservative deployment recommendations, not measured memory peaks
or guarantees. NF4 still downloads the original BF16 snapshot before loading.
"""

import os
import shutil
from pathlib import Path

from .clef_contract import RELEASES

GIB = 1024**3
SNAPSHOT_BYTES = {"clef": 54989894057, "clef-flash": 19083377402}
# Includes room for the dense lexical/schema heads and inference overhead.
MEMORY_GIB = {
    ("clef", "none", "cuda"): 80,
    ("clef", "nf4", "cuda"): 32,
    ("clef", "none", "cpu"): 80,
    ("clef", "nf4", "cpu"): 48,
    ("clef-flash", "none", "cuda"): 32,
    ("clef-flash", "nf4", "cuda"): 16,
    ("clef-flash", "none", "cpu"): 32,
    ("clef-flash", "nf4", "cpu"): 16,
}


def available_ram():
    """Available host RAM capped by remaining Linux cgroup memory, if present."""
    available = os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    root = Path("/sys/fs/cgroup")
    try:
        limit = (root / "memory.max").read_text().strip()
        if limit != "max":
            remaining = int(limit) - int((root / "memory.current").read_text())
            available = min(available, max(0, remaining))
    except (OSError, ValueError):
        pass
    return available


def inspect_resources(model, quantization, device, cache_dir=None, revision=None):
    kind = "cuda" if device.startswith("cuda") else "cpu" if device == "cpu" else None
    if kind is None:
        raise ValueError("Planning check supports cpu, cuda, or cuda:N")
    target = Path(cache_dir or os.environ.get("HF_HOME", "~/.cache/huggingface")).expanduser()
    while not target.exists():
        target = target.parent
    disk_free = shutil.disk_usage(target).free
    # Reserve 5 GiB beyond the upstream snapshot; environment is installed separately.
    snapshot = SNAPSHOT_BYTES[model] if revision in (None, RELEASES[model][1]) else None
    disk_required = snapshot + 5 * GIB if snapshot is not None else None
    memory_required = MEMORY_GIB[model, quantization, kind] * GIB
    memory_available = available_ram() if kind == "cpu" else None
    gpu_error = None
    if kind == "cuda":
        try:
            import torch

            if torch.cuda.is_available():
                memory_available, _ = torch.cuda.mem_get_info(torch.device(device))
            else:
                gpu_error = "CUDA unavailable"
        except (ImportError, RuntimeError, ValueError, AssertionError):
            gpu_error = "CUDA memory could not be inspected; install the GPU runtime"
    disk_ok = disk_required is not None and disk_free >= disk_required
    memory_ok = memory_available is not None and memory_available >= memory_required
    return {
        "model": model,
        "repository": RELEASES[model][0],
        "revision": revision or RELEASES[model][1],
        "quantization": quantization,
        "device": device,
        "snapshot_bytes": snapshot,
        "fresh_download_disk_required_bytes": disk_required,
        "disk_free_bytes": disk_free,
        "fresh_download_disk_budget_met": disk_ok,
        "recommended_available_memory_bytes": memory_required,
        "available_memory_bytes": memory_available,
        "memory_planning_budget_met": memory_ok,
        "gpu_inspection_error": gpu_error,
        "ready_for_fresh_deployment": disk_ok and memory_ok,
        "inference_tested": False,
        "notes": [
            "Planning budgets are estimates, not inference measurements or guarantees.",
            "Disk check assumes a fresh snapshot; existing cache is not deducted or verified.",
            "Custom revisions require separate snapshot-size verification.",
            "NF4 preserves dense lexical and schema heads; original weights are downloaded.",
        ],
    }
