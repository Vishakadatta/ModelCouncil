"""Interactive setup wizard. Run with: python -m setup.setup"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import httpx
from rich.console import Console
from rich.prompt import Confirm, Prompt
from rich.table import Table

from council.models import APPROVED_MODELS, TEST_ONLY_MODELS

from . import detect, plans

console = Console()
ENV_PATH = Path(".env")
ENV_DEMO_PATH = Path(".env.demo")


def main(demo: bool = False) -> int:
    console.rule("[bold cyan]Model Council Setup[/bold cyan]")

    if demo:
        return _run_demo()

    # Step 1: local or remote
    mode = Prompt.ask(
        "Run models [cyan]locally[/cyan] or on a [cyan]remote[/cyan] GPU?",
        choices=["local", "remote"],
        default="local",
    )
    base_url = "http://localhost:11434"
    if mode == "local":
        if not _ensure_local_ollama():
            return 1
    else:
        base_url = _setup_remote()
        if not base_url:
            return 1

    # Verify connectivity BEFORE pulling anything
    if not _verify_ollama(base_url):
        console.print(
            f"[bold red]Cannot reach Ollama at {base_url}[/bold red] — "
            "fix the connection and retry."
        )
        return 1

    # Step 3: detect VRAM
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

    # Step 4: build & show plans
    available_plans = plans.build_plans(vram_gb)
    if not available_plans:
        console.print(
            "[red]No approved plan fits this VRAM budget. "
            "Need at least 8GB.[/red]"
        )
        return 1

    _render_plans(vram_gb, available_plans)
    choices = [p.name for p in available_plans]
    selection = Prompt.ask(
        "Which plan?", choices=choices, default=choices[-1]
    )
    plan = next(p for p in available_plans if p.name == selection)

    # Step 5: confirm + pull
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

    # Step 6: write .env
    _write_env(ENV_PATH, base_url, plan.council, plan.judge)
    console.print(f"\n[green]Wrote {ENV_PATH}.[/green]")
    console.print(
        "Run [bold]python council.py[/bold] or [bold]make run[/bold] to ask a question."
    )
    return 0


# ---------------------------------------------------------------------------
# Demo mode
# ---------------------------------------------------------------------------

DEMO_COUNCIL = ["tinyllama", "phi3:mini"]
DEMO_JUDGE = "phi3:mini"
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
            console.print(
                f"[red]Demo refused: {m} not in TEST_ONLY_MODELS.[/red]"
            )
            return 1
    if not _pull_models(base_url, needed):
        return 1

    _write_env(
        ENV_DEMO_PATH, base_url, DEMO_COUNCIL, DEMO_JUDGE, demo=True
    )
    console.print(f"[green]Wrote {ENV_DEMO_PATH}.[/green]")

    # Run one hardcoded question end-to-end via council.py
    cmd = [sys.executable, "council.py", "--env", str(ENV_DEMO_PATH), DEMO_QUESTION]
    console.print(f"\n[dim]Running: {' '.join(cmd)}[/dim]\n")
    return subprocess.call(cmd)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ensure_local_ollama() -> bool:
    if not shutil.which("ollama"):
        console.print(
            "[red]Ollama is not installed.[/red] "
            "Install from https://ollama.com/download then re-run setup."
        )
        return False
    # Try to reach it; if not running, start it in the background.
    try:
        httpx.get("http://localhost:11434/api/tags", timeout=2.0)
        return True
    except httpx.HTTPError:
        pass
    console.print("Starting [bold]ollama serve[/bold] in the background...")
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
    console.print("[red]Ollama did not become reachable.[/red]")
    return False


def _setup_remote() -> str | None:
    kind = Prompt.ask(
        "Connect via [cyan]ssh[/cyan] tunnel or [cyan]direct[/cyan] URL?",
        choices=["ssh", "direct"],
        default="ssh",
    )
    if kind == "direct":
        url = Prompt.ask("Ollama base URL", default="http://remote:11434")
        return url

    host = Prompt.ask("SSH host")
    user = Prompt.ask("SSH user")
    port = Prompt.ask("SSH port", default="22")
    cmd = ["ssh", "-L", "11434:localhost:11434", f"{user}@{host}", "-p", port, "-N"]
    console.print(f"[dim]Opening tunnel: {' '.join(cmd)} &[/dim]")
    try:
        subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except FileNotFoundError:
        console.print("[red]ssh not found on PATH.[/red]")
        return None
    time.sleep(2)
    if not _verify_ollama("http://localhost:11434"):
        console.print(
            "[yellow]Tunnel not yet active. Run this command manually in another terminal:[/yellow]\n"
            f"  {' '.join(cmd)} &"
        )
        if not Confirm.ask("Tunnel is up now?", default=True):
            return None
    return "http://localhost:11434"


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
        title=f"Model Council — Setup Plans ({vram_gb}GB available)",
        show_lines=True,
    )
    table.add_column("Plan", style="bold cyan")
    table.add_column("Council")
    table.add_column("Judge")
    table.add_column("Peak VRAM")
    for p in plan_list:
        table.add_row(
            p.name.upper(),
            " + ".join(p.council),
            p.judge,
            f"~{p.peak_vram_gb}GB",
        )
    console.print(table)


def _pull_models(base_url: str, models: list[str]) -> bool:
    """Pull models via Ollama, streaming progress lines."""
    for m in models:
        console.print(f"\n[bold]Pulling {m}...[/bold]")
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


def _write_env(
    path: Path,
    base_url: str,
    council: list[str],
    judge: str,
    demo: bool = False,
) -> None:
    contents = (
        f"OLLAMA_BASE={base_url}\n"
        f"COUNCIL_MODELS={','.join(council)}\n"
        f"JUDGE_MODEL={judge}\n"
        f"KEEP_ALIVE=0\n"
    )
    if demo:
        contents = "# DEMO env — tiny models only\n" + contents
    path.write_text(contents)


if __name__ == "__main__":
    demo = "--demo" in sys.argv
    sys.exit(main(demo=demo))
