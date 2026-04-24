"""Dynamic model discovery from NVIDIA NIM API.

Queries https://integrate.api.nvidia.com/v1/models live, applies policy
rules, enforces geographic diversity, and returns a validated council + judge
plan. Specific model names are NEVER hardcoded — policy is entirely rule-based.

Policy rules (in order):
  1. Reject blocked publishers  (Chinese-origin companies — see NIM_BLOCKED_PUBLISHERS)
  2. Reject unknown publishers   (unknown origin = not trusted)
  3. Extract parameter count     (reject if unknown or in the 15–29B ambiguous gap)
  4. Classify by size            (council < 15B,  judge >= 30B)
  5. Geographic diversity        (council must have ≥1 USA and ≥1 non-USA model)
  6. Judge preference            (prefer non-USA judge where available)
"""

from __future__ import annotations

import re
import webbrowser
from dataclasses import dataclass

import httpx
from rich.console import Console
from rich.prompt import Confirm
from rich.table import Table

from council.models import NIM_BLOCKED_PUBLISHERS, NIM_PUBLISHER_MAP

NIM_BASE = "https://integrate.api.nvidia.com/v1"
console = Console()


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class NIMModel:
    model_id: str          # e.g. "meta/llama-3.1-8b-instruct"
    publisher: str         # e.g. "meta"
    company: str           # e.g. "Meta"
    country: str           # e.g. "USA"
    param_b: float         # billions — always set (never None after filtering)
    role: str              # "council" or "judge"

    @property
    def origin_str(self) -> str:
        return f"{self.company}, {self.country}"

    @property
    def param_str(self) -> str:
        if self.param_b >= 1:
            return f"{self.param_b:.0f}B"
        return f"{self.param_b * 1000:.0f}M"


# ---------------------------------------------------------------------------
# Publisher helpers (uses shared registry from council/models.py)
# ---------------------------------------------------------------------------

def is_blocked_origin(publisher: str) -> bool:
    """Return True if this publisher is on the blocked list."""
    return publisher.lower() in NIM_BLOCKED_PUBLISHERS


def _publisher_from_id(model_id: str) -> str:
    """Extract lowercase publisher prefix from a NIM model ID."""
    if "/" in model_id:
        return model_id.split("/", 1)[0].lower()
    # Fallback for bare model names like "llama3:8b"
    return model_id.split(":")[0].lower()


def _company_country(publisher: str) -> tuple[str, str] | None:
    """
    Map publisher key to (company, country).
    Returns None for blocked or unknown publishers.
    Per policy: unknown origin = reject.
    """
    pub = publisher.lower()
    if pub in NIM_BLOCKED_PUBLISHERS:
        return None  # explicitly blocked
    entry = NIM_PUBLISHER_MAP.get(pub)
    if entry:
        return entry
    return None  # unknown = reject


# ---------------------------------------------------------------------------
# Parameter-count extraction
# ---------------------------------------------------------------------------

def _extract_param_b(model_id: str) -> float | None:
    """
    Parse parameter count (in billions) from a NIM model ID string.

    Handles:
      - Standard:    llama-3.1-8b-instruct  → 8.0
      - Fractional:  phi-3-mini-3.8b        → 3.8
      - Sub-billion: qwen-0.5b              → 0.5
      - MoE:         mixtral-8x7b           → 56.0
    """
    name = model_id.lower()

    # Mixture-of-Experts: 8x7b → 8 * 7 = 56B
    moe = re.search(r"(\d+)x(\d+)b", name)
    if moe:
        return float(moe.group(1)) * float(moe.group(2))

    # Standard with separator before/after: -8b-, _3.8b_, .0.5b
    sep_match = re.search(r"[\-_\.](\d+(?:\.\d+)?)b(?:[\-_\.]|$)", name)
    if sep_match:
        return float(sep_match.group(1))

    # Bare token: anything like "8b" not preceded or followed by a digit
    bare_match = re.search(r"(?<!\d)(\d+(?:\.\d+)?)b(?!\w)", name)
    if bare_match:
        return float(bare_match.group(1))

    return None


def _classify_role(param_b: float) -> str | None:
    """
    council:  param_b < 15
    judge:    param_b >= 30
    Returns None for 15–29B (ambiguous gap — skip).
    """
    if param_b < 15:
        return "council"
    if param_b >= 30:
        return "judge"
    return None  # 15–29B: skip


# ---------------------------------------------------------------------------
# Core fetch + filter
# ---------------------------------------------------------------------------

def fetch_nim_models(api_key: str) -> list[NIMModel]:
    """
    GET /v1/models from NIM, apply all policy rules, return usable models.

    Raises httpx.HTTPError on network / auth failure.
    """
    r = httpx.get(
        f"{NIM_BASE}/models",
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=15.0,
    )
    r.raise_for_status()

    raw_models: list[dict] = r.json().get("data", [])
    usable: list[NIMModel] = []
    n_filtered = 0

    for m in raw_models:
        model_id: str = m.get("id", "").strip()
        if not model_id:
            continue

        publisher = _publisher_from_id(model_id)

        # Rule 1 — blocked origin
        if is_blocked_origin(publisher):
            n_filtered += 1
            continue

        # Rule 2 — unknown origin
        cc = _company_country(publisher)
        if cc is None:
            n_filtered += 1
            continue
        company, country = cc

        # Rule 3 — parameter count
        param_b = _extract_param_b(model_id)
        if param_b is None:
            n_filtered += 1
            continue

        # Rule 4 — role classification (rejects 15–29B gap)
        role = _classify_role(param_b)
        if role is None:
            n_filtered += 1
            continue

        usable.append(NIMModel(
            model_id=model_id,
            publisher=publisher,
            company=company,
            country=country,
            param_b=param_b,
            role=role,
        ))

    if n_filtered:
        console.print(
            f"[dim]  Filtered {n_filtered} model(s): "
            f"blocked origin, unknown origin, unknown size, or 15–29B gap.[/dim]"
        )

    return usable


# ---------------------------------------------------------------------------
# Diversity selection
# ---------------------------------------------------------------------------

def _apply_diversity(candidates: list[NIMModel]) -> list[NIMModel]:
    """
    Select up to 3 council members satisfying geographic diversity:
      - At least one USA-origin model.
      - At least one non-USA model (if any exist after filtering).
      - No duplicate publishers.

    Strategy: prioritise diversity first, then fill remaining slots.
    """
    usa_models = [m for m in candidates if m.country == "USA"]
    non_usa_models = [m for m in candidates if m.country != "USA"]

    if not usa_models:
        return []  # cannot satisfy "at least one USA" rule

    chosen: list[NIMModel] = []
    seen_publishers: set[str] = set()

    # Slot 1: take the first non-USA model (diversity anchor)
    for m in non_usa_models:
        if m.publisher not in seen_publishers:
            chosen.append(m)
            seen_publishers.add(m.publisher)
            break

    # Slots 2–3: fill from USA models (then any remaining non-USA)
    fill_order = usa_models + non_usa_models
    for m in fill_order:
        if len(chosen) >= 3:
            break
        if m.publisher not in seen_publishers:
            chosen.append(m)
            seen_publishers.add(m.publisher)

    return chosen


def _pick_judge(candidates: list[NIMModel]) -> NIMModel | None:
    """
    Pick the largest judge. Prefer non-USA for geographic diversity.
    Falls back to largest USA judge if no non-USA candidate exists.
    """
    if not candidates:
        return None

    by_size = sorted(candidates, key=lambda m: m.param_b, reverse=True)
    non_usa = [j for j in by_size if j.country != "USA"]
    return non_usa[0] if non_usa else by_size[0]


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------

def _show_plan(council: list[NIMModel], judge: NIMModel) -> None:
    """Render the discovered plan in a Rich table."""
    table = Table(
        title="Available Model Council Plan  (NVIDIA NIM — live discovery)",
        show_lines=True,
    )
    table.add_column("Role",          style="bold cyan", min_width=8)
    table.add_column("Model ID",      min_width=38)
    table.add_column("Company",       min_width=14)
    table.add_column("Country",       min_width=12)
    table.add_column("Params",        min_width=7)

    for m in council:
        table.add_row("Council", m.model_id, m.company, m.country, m.param_str)
    table.add_row(
        "Judge", judge.model_id, judge.company, judge.country, judge.param_str,
        style="bold yellow",
    )
    console.print(table)

    # Diversity summary
    countries = sorted({m.country for m in council})
    has_usa = any(m.country == "USA" for m in council)
    has_non_usa = any(m.country != "USA" for m in council)

    if has_usa and has_non_usa:
        console.print(
            f"[green]Geographic diversity: council covers {', '.join(countries)} ✓[/green]"
        )
    else:
        console.print(
            f"[yellow]Geographic diversity: council covers {', '.join(countries)} "
            f"(only one country available after filtering)[/yellow]"
        )
    console.print("[green]All models: non-Chinese origin verified ✓[/green]")


def _list_alternatives(
    council_candidates: list[NIMModel],
    judge_candidates: list[NIMModel],
) -> None:
    """Print all available candidates so the user can re-run manually."""
    console.print("\n[cyan]All available council candidates (<15B, non-blocked):[/cyan]")
    for m in council_candidates:
        console.print(f"  • {m.model_id:<42} {m.origin_str}  {m.param_str}")

    console.print("\n[cyan]All available judge candidates (≥30B, non-blocked):[/cyan]")
    for m in judge_candidates:
        console.print(f"  • {m.model_id:<42} {m.origin_str}  {m.param_str}")

    console.print(
        "\n[dim]Re-run `make setup` to try again; "
        "the wizard always selects automatically from the live catalogue.[/dim]"
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def discover_and_plan(api_key: str) -> tuple[list[str], str] | None:
    """
    Full discovery flow. Queries NIM, filters, selects, shows plan, asks user.

    Returns:
        (council_model_ids, judge_model_id)  on success.
        None                                  if cancelled or no models found.

    Raises:
        httpx.HTTPError on unrecoverable network failure.
    """
    console.print("\nFetching available models from NVIDIA NIM…")

    try:
        models = fetch_nim_models(api_key)
    except httpx.HTTPStatusError as e:
        console.print(
            f"[red]NIM API returned {e.response.status_code}. "
            f"Check your API key and try again.[/red]"
        )
        return None
    except httpx.HTTPError as e:
        console.print(f"[red]Network error reaching NIM: {e}[/red]")
        return None

    council_candidates = [m for m in models if m.role == "council"]
    judge_candidates   = [m for m in models if m.role == "judge"]

    if not council_candidates:
        console.print(
            "[red]No council-eligible models (<15B, non-blocked) found.\n"
            "Visit https://build.nvidia.com to check current catalogue.[/red]"
        )
        return None

    if not judge_candidates:
        console.print(
            "[red]No judge-eligible models (≥30B, non-blocked) found.\n"
            "Visit https://build.nvidia.com to check current catalogue.[/red]"
        )
        return None

    council = _apply_diversity(council_candidates)
    judge   = _pick_judge(judge_candidates)

    if len(council) < 2:
        console.print(
            "[red]Could not form a diverse council of at least 2 models "
            "after applying geographic diversity rules.[/red]"
        )
        _list_alternatives(council_candidates, judge_candidates)
        return None

    # Interactive confirm loop
    while True:
        console.print()
        _show_plan(council, judge)

        if Confirm.ask("\nProceed with this plan?", default=True):
            return [m.model_id for m in council], judge.model_id

        # User said no — show alternatives and let them decide
        _list_alternatives(council_candidates, judge_candidates)

        if not Confirm.ask(
            "\nThe wizard always auto-selects from live results. "
            "Accept this plan anyway?",
            default=False,
        ):
            console.print("Setup cancelled.")
            return None
