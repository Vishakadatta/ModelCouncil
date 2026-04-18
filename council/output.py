"""Rich terminal output for the council pipeline."""

from __future__ import annotations

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.text import Text

from .orchestrator import CouncilTurn

console = Console()


def banner_council_start(models: list[str]) -> None:
    console.print(
        f"\n[bold cyan][Council][/bold cyan] Sending prompt to "
        f"{len(models)} models simultaneously: "
        f"[dim]{', '.join(models)}[/dim]\n"
    )


def render_council_turn(turn: CouncilTurn) -> None:
    body = Text(turn.response.strip() + "\n\n")
    body.append(
        f"[done in {turn.latency_seconds}s] [VRAM purged]",
        style="dim green",
    )
    console.print(
        Panel(
            body,
            title=f"[bold]{turn.model}[/bold] ([italic]{turn.origin}[/italic])",
            border_style="cyan",
            expand=True,
        )
    )


def banner_judge_loading(model: str) -> None:
    console.print(
        f"\n[bold magenta][Judge][/bold magenta] Loading [bold]{model}[/bold]..."
    )
    console.print(
        f"[bold magenta][Judge][/bold magenta] Synthesizing final answer...\n"
    )


def stream_judge(judge_model: str):
    """Context manager yielding a callback that appends streamed judge text."""
    text = Text()
    panel = Panel(
        text,
        title=f"[bold]Final Answer ({judge_model})[/bold]",
        border_style="magenta",
        expand=True,
    )
    live = Live(panel, console=console, refresh_per_second=12)

    class _Streamer:
        def __enter__(self_inner):
            live.__enter__()
            return self_inner.append

        def __exit__(self_inner, exc_type, exc, tb):
            live.__exit__(exc_type, exc, tb)

        @staticmethod
        def append(chunk: str) -> None:
            text.append(chunk)
            live.update(panel)

    return _Streamer()


def saved(path) -> None:
    console.print(f"\n[green]Saved:[/green] {path}\n")


def policy_error(msg: str) -> None:
    console.print(f"[bold red]Policy error:[/bold red] {msg}")
