"""Cross-platform VRAM detection.

Tries the most reliable per-OS source first, then falls back. Always returns
either an integer GB value or None — never raises. Caller decides whether to
ask the user manually.
"""

from __future__ import annotations

import platform
import re
import shutil
import subprocess


def _run(cmd: list[str], timeout: int = 5) -> str | None:
    try:
        out = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, check=False
        )
        return out.stdout if out.returncode == 0 else None
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None


def _nvidia_smi() -> int | None:
    if not shutil.which("nvidia-smi"):
        return None
    out = _run([
        "nvidia-smi",
        "--query-gpu=memory.total",
        "--format=csv,noheader,nounits",
    ])
    if not out:
        return None
    try:
        mb = max(int(line.strip()) for line in out.splitlines() if line.strip())
        return round(mb / 1024)
    except ValueError:
        return None


def _rocm_smi() -> int | None:
    if not shutil.which("rocm-smi"):
        return None
    out = _run(["rocm-smi", "--showmeminfo", "vram"])
    if not out:
        return None
    m = re.search(r"Total\s*Memory.*?:\s*(\d+)", out)
    if not m:
        return None
    return round(int(m.group(1)) / (1024 ** 3))


def _macos_unified_memory() -> int | None:
    """Apple Silicon: unified memory acts as VRAM. Use total RAM as a proxy."""
    if platform.system() != "Darwin":
        return None
    out = _run(["sysctl", "-n", "hw.memsize"])
    if not out:
        return None
    try:
        return round(int(out.strip()) / (1024 ** 3))
    except ValueError:
        return None


def _proc_meminfo() -> int | None:
    if platform.system() != "Linux":
        return None
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    kb = int(line.split()[1])
                    return round(kb / (1024 ** 2))
    except OSError:
        return None
    return None


def detect_vram_gb() -> tuple[int | None, str]:
    """Return (gb, source) — source is a label for display."""
    for fn, label in (
        (_nvidia_smi, "nvidia-smi"),
        (_rocm_smi, "rocm-smi"),
        (_macos_unified_memory, "macOS unified memory (sysctl)"),
        (_proc_meminfo, "/proc/meminfo"),
    ):
        gb = fn()
        if gb:
            return gb, label
    return None, "auto-detection failed"


MANUAL_HINTS = """
Manual VRAM check commands:
  Windows:   nvidia-smi  (or Task Manager -> Performance -> GPU)
  macOS:     system_profiler SPDisplaysDataType | grep VRAM
             ioreg -l | grep "VRAM,totalMB"
  Linux:     nvidia-smi --query-gpu=memory.total --format=csv
             cat /proc/driver/nvidia/gpus/*/information
""".strip()
