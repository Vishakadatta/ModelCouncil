"""Build Safe / Balanced / Max plans from the approved model list.

Rules enforced:
  - Council must use diverse vendors (no two from the same origin).
  - Plan peak VRAM = max(sum of council vram, judge vram).
    keep_alive=0 means council VRAM is freed before judge loads, so peak
    is whichever is larger — not the sum of both.
  - Never propose a plan whose peak exceeds available VRAM.
"""

from __future__ import annotations

from dataclasses import dataclass

from council.models import APPROVED_MODELS, ModelSpec, origin_for, vram_for


@dataclass
class Plan:
    name: str
    council: list[str]
    judge: str
    peak_vram_gb: int

    def summary(self) -> str:
        c = " + ".join(self.council)
        return f"Council: {c}\nJudge:   {self.judge}\nVRAM:    ~{self.peak_vram_gb}GB peak"


def _peak(council: list[str], judge: str) -> int:
    council_sum = sum(vram_for(m) for m in council)
    return max(council_sum, vram_for(judge))


def _diverse(tags: list[str]) -> bool:
    origins = [origin_for(t) for t in tags]
    return len(origins) == len(set(origins))


def _judge_candidates() -> list[ModelSpec]:
    return [m for m in APPROVED_MODELS if m.role == "judge"]


def _council_candidates(max_size: int) -> list[ModelSpec]:
    return [m for m in APPROVED_MODELS if m.role == "council" and m.vram_gb <= max_size]


def build_plans(available_gb: int) -> list[Plan]:
    """Return up to 3 plans that fit in available_gb, ordered safe -> max."""
    plans: list[Plan] = []

    # Tier policy from spec:
    #   8GB  -> Safe only
    #   16GB -> Safe + Balanced
    #   24GB+ -> all three

    safe = _try_plan(
        name="safe",
        council_size=2,
        council_pool=["phi3:mini", "mistral:7b", "llama3.1:8b"],
        judge_pool=["llama3.1:8b", "mistral:7b"],
        available_gb=available_gb,
    )
    if safe:
        plans.append(safe)

    if available_gb >= 16:
        balanced = _try_plan(
            name="balanced",
            council_size=3,
            council_pool=["llama3.1:8b", "mistral:7b", "gemma2:9b", "command-r"],
            judge_pool=["gemma2:27b", "llama3.1:8b"],
            available_gb=available_gb,
        )
        if balanced:
            plans.append(balanced)

    if available_gb >= 24:
        mx = _try_plan(
            name="max",
            council_size=3,
            council_pool=["llama3.1:8b", "mistral:7b", "command-r", "gemma2:9b"],
            judge_pool=["command-r-plus", "gemma2:27b"],
            available_gb=available_gb,
        )
        if mx:
            plans.append(mx)

    return plans


def _try_plan(
    name: str,
    council_size: int,
    council_pool: list[str],
    judge_pool: list[str],
    available_gb: int,
) -> Plan | None:
    # Pick the largest diverse council that fits, preferring earlier entries.
    council = _pick_diverse(council_pool, council_size)
    if not council:
        return None
    for judge in judge_pool:
        peak = _peak(council, judge)
        if peak <= available_gb:
            return Plan(name=name, council=council, judge=judge, peak_vram_gb=peak)
    return None


def _pick_diverse(pool: list[str], size: int) -> list[str] | None:
    chosen: list[str] = []
    seen_origins: set[str] = set()
    for tag in pool:
        origin = origin_for(tag)
        if origin in seen_origins:
            continue
        chosen.append(tag)
        seen_origins.add(origin)
        if len(chosen) == size:
            return chosen
    return chosen if len(chosen) >= 2 else None


def estimate_download_gb(plan: Plan) -> int:
    """Rough on-disk size = vram_gb sum (close enough for a confirmation prompt)."""
    return sum(vram_for(m) for m in plan.council) + vram_for(plan.judge)
