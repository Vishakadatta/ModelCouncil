"""Interactive setup wizard. Run with: python -m setup.setup

First question is always backend selection:
  1. Local GPU  (Ollama on this machine)
  2. NVIDIA NIM (free API — no GPU needed)

LOCAL path  → existing VRAM detection + plan flow (unchanged).
NIM path    → API key prompt + nim_discover.discover_and_plan().
"""

from __future__ import annotations

import json
import platform
import shutil
import subprocess
import sys
import time
import webbrowser
from pathlib import Path

import httpx
from rich.console import Console
from rich.prompt import Confirm, Prompt
from rich.table import Table

from council.models import APPROVED_MODELS, TEST_ONLY_MODELS
from council.nim_client import validate_key as _nim_validate_key

from . import detect, plans
from .nim_discover import discover_and_plan as _nim_discover

console = Console()
ENV_PATH      = Path(".env")
ENV_DEMO_PATH = Path(".env.demo")


def main(demo: bool = False) -> int:
    console.rule("[bold cyan]Model Council Setup[/bold cyan]")

    if demo:
        return _run_demo()

    # ── Step 1: backend selection ────────────────────────────────────────────
    console.print(
        "\n[bold]How do you want to run the AI models?[/bold]\n\n"
        "  [cyan]1.[/cyan]  Local GPU   (Ollama on this machine)\n"
        "  [cyan]2.[/cyan]  NVIDIA NIM  (free API — no GPU needed)\n"
    )
    backend_choice = Prompt.ask("Choice", choices=["1", "2"], default="1")

    if backend_choice == "2":
        return _run_nim_setup()
    else:
        return _run_local_setup()


# ---------------------------------------------------------------------------
# LOCAL (Ollama) setup path
# ---------------------------------------------------------------------------

def _run_local_setup() -> int:
    if not _ensure_local_ollama():
        return 1

    base_url = "http://localhost:11434"

    if not _verify_ollama(base_url):
        console.print(
            f"[bold red]Cannot reach Ollama at {base_url}[/bold red] — "
            "fix the connection and retry."
        )
        return 1

    # Detect VRAM
    vram_gb, source = detect.detect_vram_gb()
    if vram_gb:
        console.print(f"\nDetected [bold]{vram_gb} GB[/bold] via {source}.")
    else:
        console.print(f"\n[yellow]{source}.[/yellow]")
        console.print(detect.MANUAL_HINTS)
        vram_gb = int(
            Prompt.ask(
                "How much VRAM/unified memory do you want to plan for? (GB)",
                default="16",
            )
        )

    # Build + show plans
    available_plans = plans.build_plans(vram_gb)
    if not available_plans:
        console.print(
            "[red]No approved plan fits this VRAM budget. Need at least 8 GB.[/red]"
        )
        return 1

    _render_plans(vram_gb, available_plans)
    choices = [p.name for p in available_plans]
    selection = Prompt.ask("Which plan?", choices=choices, default=choices[-1])
    plan = next(p for p in available_plans if p.name == selection)

    # Confirm download
    download_gb = plans.estimate_download_gb(plan)
    if not Confirm.ask(
        f"\nThis will download approximately [bold]{download_gb} GB[/bold]. Continue?",
        default=True,
    ):
        console.print("Aborted.")
        return 0

    needed = list(dict.fromkeys(plan.council + [plan.judge]))
    if not _pull_models(base_url, needed):
        return 1

    _write_env_ollama(ENV_PATH, base_url, plan.council, plan.judge)
    console.print(f"\n[green]Wrote {ENV_PATH}.[/green]")
    console.print(
        "Run [bold]make web[/bold] or [bold]python council.py[/bold] to ask a question."
    )
    return 0


# ---------------------------------------------------------------------------
# NIM setup path
# ---------------------------------------------------------------------------

def _run_nim_setup() -> int:
    console.print(
        "\n[bold]NVIDIA NIM[/bold] runs powerful AI models on NVIDIA's servers.\n"
        "You need a free API key from [cyan]build.nvidia.com[/cyan]\n\n"
        "To get your key:\n"
        "  1. Go to [cyan]https://build.nvidia.com[/cyan]\n"
        "  2. Sign in or create a free account\n"
        "  3. Click your profile icon (top right)\n"
        "  4. Click [bold]API Keys → Generate API Key[/bold]\n"
        "  5. Copy the key (starts with [bold]nvapi-[/bold])\n"
    )

    api_key = ""
    while True:
        api_key = Prompt.ask("Paste your API key here (or press Enter to open browser)").strip()

        if not api_key:
            console.print("Opening [cyan]https://build.nvidia.com[/cyan] in your browser…")
            webbrowser.open("https://build.nvidia.com")
            console.print("Waiting for your API key…\n")
            continue

        # Validate
        console.print("Validating API key…")
        try:
            ok = _nim_validate_key(api_key)
        except httpx.HTTPStatusError as e:
            if e.response.status_code in (401, 403):
                console.print(
                    "[red]Invalid API key (401 Unauthorized). "
                    "Check the key and try again.[/red]"
                )
                api_key = ""
                continue
            console.print(f"[red]Unexpected HTTP error {e.response.status_code}: {e}[/red]")
            if not Confirm.ask("Retry?", default=True):
                return 1
            api_key = ""
            continue
        except httpx.HTTPError as e:
            console.print(f"[red]Network error: {e}[/red]")
            if not Confirm.ask("Retry?", default=True):
                return 1
            api_key = ""
            continue

        if not ok:
            console.print("[red]Key validation failed. Try again.[/red]")
            api_key = ""
            continue

        console.print("[green]API key valid. Fetching available models…[/green]")
        break

    # Dynamic discovery
    result = _nim_discover(api_key)
    if result is None:
        return 1
    council, judge = result

    _write_env_nim(ENV_PATH, api_key, council, judge)
    console.print(f"\n[green]Wrote {ENV_PATH}.[/green]")
    console.print(
        "Run [bold]make web[/bold] to start the web UI.\n"
        "[dim]The server reads NVIDIA_API_KEY from .env — do not commit that file.[/dim]"
    )
    return 0


# ---------------------------------------------------------------------------
# Demo mode (unchanged)
# ---------------------------------------------------------------------------

DEMO_COUNCIL  = ["tinyllama", "phi3:mini"]
DEMO_JUDGE    = "phi3:mini"
DEMO_QUESTION = "In two sentences, what is gravity?"


def _run_demo() -> int:
    console.print(
        "[bold yellow]DEMO MODE — using tiny models, "
        "not suitable for real use.[/bold yellow]\n"
    )
    base_url = "http://localhost:11434"
    if not _ensure_local_ollama():
        return 1
    if not _verify_ollama(base_url):
        return 1

    needed = list(dict.fromkeys(DEMO_COUNCIL + [DEMO_JUDGE]))
    for m in needed:
        if m not in TEST_ONLY_MODELS:
            console.print(f"[red]Demo refused: {m} not in TEST_ONLY_MODELS.[/red]")
            return 1
    if not _pull_models(base_url, needed):
        return 1

    _write_env_ollama(ENV_DEMO_PATH, base_url, DEMO_COUNCIL, DEMO_JUDGE, demo=True)
    console.print(f"[green]Wrote {ENV_DEMO_PATH}.[/green]")

    cmd = [sys.executable, "council.py", "--env", str(ENV_DEMO_PATH), DEMO_QUESTION]
    console.print(f"\n[dim]Running: {' '.join(cmd)}[/dim]\n")
    return subprocess.call(cmd)


# ---------------------------------------------------------------------------
# Ollama helpers
# ---------------------------------------------------------------------------

def _ensure_local_ollama() -> bool:
    """Make sure Ollama is installed and running. Offers to install if missing."""
    if not shutil.which("ollama"):
        console.print("[red]Ollama is not installed.[/red]")
        if not Confirm.ask("Install Ollama now?", default=True):
            console.print(
                "Install manually from [cyan]https://ollama.com/download[/cyan] "
                "then re-run setup."
            )
            return False

        console.print("\nInstalling Ollama…")
        try:
            if platform.system() == "Darwin":
                # macOS — prefer Homebrew, fall back to the install script
                if shutil.which("brew"):
                    subprocess.run(["brew", "install", "ollama"], check=True)
                else:
                    subprocess.run(
                        "curl -fsSL https://ollama.com/install.sh | sh",
                        shell=True, check=True,
                    )
            else:
                # Linux
                subprocess.run(
                    "curl -fsSL https://ollama.com/install.sh | sh",
                    shell=True, check=True,
                )
        except subprocess.CalledProcessError as e:
            console.print(f"[red]Installation failed: {e}[/red]")
            console.print(
                "Install manually from [cyan]https://ollama.com/download[/cyan]"
            )
            return False

        # Verify install succeeded
        if not shutil.which("ollama"):
            console.print(
                "[red]ollama binary not found after install. "
                "Open a new terminal and re-run setup.[/red]"
            )
            return False
        console.print("[green]Ollama installed.[/green]")

    # Try to reach it; start it if not running
    try:
        httpx.get("http://localhost:11434/api/tags", timeout=2.0)
        return True
    except httpx.HTTPError:
        pass

    console.print("Starting [bold]ollama serve[/bold] in the background…")
    subprocess.Popen(
        ["ollama", "serve"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    for _ in range(20):
        time.sleep(0.5)
        try:
            httpx.get("http://localhost:11434/api/tags", timeout=2.0)
            return True
        except httpx.HTTPError:
            continue

    console.print("[red]Ollama did not become reachable within 10 s.[/red]")
    return False


def _verify_ollama(base_url: str) -> bool:
    try:
        r = httpx.get(f"{base_url.rstrip('/')}/api/tags", timeout=5.0)
        r.raise_for_status()
        return True
    except httpx.HTTPError as e:
        console.print(f"[red]Ollama check failed: {e}[/red]")
        return False


def _render_plans(vram_gb: int, plan_list) -> None:
    table = Table(
        title=f"Model Council — Setup Plans ({vram_gb} GB available)",
        show_lines=True,
    )
    table.add_column("Plan",      style="bold cyan")
    table.add_column("Council")
    table.add_column("Judge")
    table.add_column("Peak VRAM")
    for p in plan_list:
        table.add_row(
            p.name.upper(),
            " + ".join(p.council),
            p.judge,
            f"~{p.peak_vram_gb} GB",
        )
    console.print(table)


def _pull_models(base_url: str, models: list[str]) -> bool:
    for m in models:
        console.print(f"\n[bold]Pulling {m}…[/bold]")
        try:
            with httpx.stream(
                "POST",
                f"{base_url.rstrip('/')}/api/pull",
                json={"model": m, "stream": True},
                timeout=None,
            ) as r:
                r.raise_for_status()
                last_status = ""
                for line in r.iter_lines():
                    if not line:
                        continue
                    obj = json.loads(line)
                    status = obj.get("status", "")
                    if status and status != last_status:
                        console.print(f"  {status}")
                        last_status = status
                    if obj.get("error"):
                        console.print(f"[red]Error pulling {m}: {obj['error']}[/red]")
                        return False
        except httpx.HTTPError as e:
            console.print(f"[red]Failed to pull {m}: {e}[/red]")
            return False
    return True


# ---------------------------------------------------------------------------
# .env writers
# ---------------------------------------------------------------------------

def _write_env_ollama(
    path: Path,
    base_url: str,
    council: list[str],
    judge: str,
    demo: bool = False,
) -> None:
    lines = [
        "# Generated by `make setup` — do not commit this file.",
        f"BACKEND=ollama",
        f"OLLAMA_BASE={base_url}",
        f"COUNCIL_MODELS={','.join(council)}",
        f"JUDGE_MODEL={judge}",
        "# keep_alive=0 forces each council model to release VRAM immediately.",
        "KEEP_ALIVE=0",
    ]
    if demo:
        lines = ["# DEMO env — tiny models only"] + lines
    path.write_text("\n".join(lines) + "\n")


def _write_env_nim(
    path: Path,
    api_key: str,
    council: list[str],
    judge: str,
) -> None:
    lines = [
        "# Generated by `make setup` — do not commit this file.",
        "# NVIDIA_API_KEY is a secret — never push this file to git.",
        "BACKEND=nim",
        f"NVIDIA_API_KEY={api_key}",
        "NIM_BASE=https://integrate.api.nvidia.com/v1",
        f"COUNCIL_MODELS={','.join(council)}",
        f"JUDGE_MODEL={judge}",
    ]
    path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    demo = "--demo" in sys.argv
    sys.exit(main(demo=demo))
