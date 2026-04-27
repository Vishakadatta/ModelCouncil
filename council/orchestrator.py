"""Async council + judge orchestration.

Architectural rules — DO NOT CHANGE:
  1. Every council generate call MUST include keep_alive=0 on the OLLAMA path.
     That flag forces Ollama to release VRAM immediately so the judge can load.
  2. The judge two-phase logic (independent answer → deliberation) lives in
     api/server.py and must not be touched here.
  3. The NIM path never sends keep_alive — that is an Ollama-only concept.

Backend routing:
  BACKEND=ollama (default) → Ollama /api/generate  (keep_alive honoured)
  BACKEND=nim              → NIM /v1/chat/completions via council.nim_client
                             (_stream_generate reads BACKEND lazily from env
                              so the .env file is always loaded first)
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import AsyncIterator, Iterable

import httpx

from .models import (
    assert_council_diverse,
    assert_production_allowed,
    origin_for,
)

# keep_alive=0 is non-negotiable for Ollama council calls. Don't change this.
COUNCIL_KEEP_ALIVE = 0
DEFAULT_TIMEOUT    = 600.0


@dataclass
class CouncilTurn:
    model: str
    origin: str
    response: str
    latency_seconds: float
    tokens_generated: int


@dataclass
class JudgeTurn:
    model: str
    origin: str
    response: str
    latency_seconds: float
    tokens_generated: int


@dataclass
class Session:
    timestamp: str
    prompt: str
    plan: str
    council: list[CouncilTurn] = field(default_factory=list)
    judge: JudgeTurn | None = None
    total_seconds: float = 0.0

    def to_dict(self) -> dict:
        return {
            "timestamp":     self.timestamp,
            "prompt":        self.prompt,
            "plan":          self.plan,
            "council":       [asdict(t) for t in self.council],
            "judge":         asdict(self.judge) if self.judge else None,
            "total_seconds": round(self.total_seconds, 2),
        }


class Orchestrator:
    def __init__(
        self,
        council_models: list[str],
        judge_model: str,
        ollama_base: str = "http://localhost:11434",
        plan_name: str = "custom",
        production: bool = True,
    ):
        self.council_models = council_models
        self.judge_model    = judge_model
        self.base           = ollama_base.rstrip("/")
        self.plan_name      = plan_name

        if production:
            assert_council_diverse(council_models)
            # Ollama static allowlist check — skipped for NIM backend because
            # nim_discover.py already applied origin + size policy at setup time.
            backend = os.environ.get("BACKEND", "ollama")
            if backend != "nim":
                for tag in council_models:
                    assert_production_allowed(tag, "council")
                assert_production_allowed(judge_model, "judge")

    # ----- HTTP helpers ---------------------------------------------------

    async def _generate(
        self,
        client: httpx.AsyncClient,
        model: str,
        prompt: str,
        keep_alive: int | str,
    ) -> tuple[str, int, float]:
        """Non-streaming generate (Ollama path only). Returns (text, tokens, latency)."""
        payload = {
            "model":      model,
            "prompt":     prompt,
            "stream":     False,
            "keep_alive": keep_alive,  # MUST be 0 for council calls
        }
        t0 = time.perf_counter()
        r  = await client.post(f"{self.base}/api/generate", json=payload)
        r.raise_for_status()
        data    = r.json()
        latency = time.perf_counter() - t0
        return (
            data.get("response", "").strip(),
            int(data.get("eval_count", 0)),
            latency,
        )

    async def _stream_generate(
        self,
        client: httpx.AsyncClient,
        model: str,
        prompt: str,
        keep_alive: int | str,
    ) -> AsyncIterator[tuple[str, dict]]:
        """
        Stream generate. Yields (chunk_text, raw_dict) per line.

        BACKEND=ollama → Ollama /api/generate  (keep_alive respected)
        BACKEND=nim    → NIM via nim_client     (keep_alive ignored — NIM concept)

        raw_dict always has keys: response, done, eval_count.
        This contract is what api/server.py depends on — do not change it.
        """
        backend = os.environ.get("BACKEND", "ollama")

        if backend == "nim":
            # NIM path — keep_alive is an Ollama-only concept, not sent.
            # Pass the appropriate fallback pool based on whether this is a
            # council model or the judge.
            from council import nim_client
            pool = (
                nim_client.JUDGE_FALLBACK_POOL
                if model == self.judge_model
                else nim_client.COUNCIL_FALLBACK_POOL
            )
            async for chunk in nim_client.stream_generate(
                model, prompt, fallback_pool=pool
            ):
                yield chunk
            return

        # Ollama path (original — keep_alive=0 for council is preserved here)
        payload = {
            "model":      model,
            "prompt":     prompt,
            "stream":     True,
            "keep_alive": keep_alive,
        }
        async with client.stream(
            "POST", f"{self.base}/api/generate", json=payload
        ) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line:
                    continue
                obj = json.loads(line)
                yield obj.get("response", ""), obj

    # ----- Council & judge -----------------------------------------------

    async def run_council(
        self, prompt: str, on_done=None
    ) -> list[CouncilTurn]:
        """Fire all council models in parallel; each releases VRAM on finish."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            async def one(model: str) -> CouncilTurn:
                text, tokens, latency = await self._generate(
                    client, model, prompt, keep_alive=COUNCIL_KEEP_ALIVE
                )
                turn = CouncilTurn(
                    model=model,
                    origin=origin_for(model),
                    response=text,
                    latency_seconds=round(latency, 2),
                    tokens_generated=tokens,
                )
                if on_done:
                    on_done(turn)
                return turn

            return await asyncio.gather(*(one(m) for m in self.council_models))

    async def run_judge_stream(
        self,
        prompt: str,
        council_turns: list[CouncilTurn],
        on_chunk=None,
    ) -> JudgeTurn:
        """Load judge, stream synthesis. Judge keeps default keep_alive."""
        synthesis_prompt = _build_judge_prompt(prompt, council_turns)
        chunks: list[str] = []
        tokens = 0
        t0 = time.perf_counter()
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            async for text, raw in self._stream_generate(
                client, self.judge_model, synthesis_prompt, keep_alive="5m"
            ):
                if text:
                    chunks.append(text)
                    if on_chunk:
                        on_chunk(text)
                if raw.get("done"):
                    tokens = int(raw.get("eval_count", 0))
        latency = time.perf_counter() - t0
        return JudgeTurn(
            model=self.judge_model,
            origin=origin_for(self.judge_model),
            response="".join(chunks).strip(),
            latency_seconds=round(latency, 2),
            tokens_generated=tokens,
        )

    # ----- Top-level session ---------------------------------------------

    async def run_session(
        self,
        prompt: str,
        results_dir: Path,
        on_council_done=None,
        on_judge_chunk=None,
        on_judge_start=None,
    ) -> tuple[Session, Path]:
        session = Session(
            timestamp=datetime.now().isoformat(timespec="seconds"),
            prompt=prompt,
            plan=self.plan_name,
        )
        t0 = time.perf_counter()
        session.council = await self.run_council(prompt, on_done=on_council_done)
        if on_judge_start:
            on_judge_start(self.judge_model)
        session.judge = await self.run_judge_stream(
            prompt, session.council, on_chunk=on_judge_chunk
        )
        session.total_seconds = time.perf_counter() - t0

        results_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out   = results_dir / f"session_{stamp}.json"
        out.write_text(json.dumps(session.to_dict(), indent=2))
        return session, out


def _build_judge_prompt(question: str, turns: Iterable[CouncilTurn]) -> str:
    parts = [
        "You are a senior judge. Three smaller models independently answered "
        "the user's question. Read all responses, weigh their merits, and "
        "synthesize the single best, most accurate final answer. Do not just "
        "pick one — combine the strongest reasoning from each.",
        "",
        f"User question: {question}",
        "",
    ]
    for i, t in enumerate(turns, 1):
        parts.append(f"--- Council answer {i} from {t.model} ({t.origin}) ---")
        parts.append(t.response)
        parts.append("")
    parts.append("Now write the final synthesized answer for the user:")
    return "\n".join(parts)


def load_env_config() -> tuple[list[str], str, str]:
    """
    Read COUNCIL_MODELS, JUDGE_MODEL, and the backend base URL from env / .env.

    For BACKEND=ollama: base = OLLAMA_BASE (default http://localhost:11434)
    For BACKEND=nim:    base = NIM_BASE    (default https://integrate.api.nvidia.com/v1)
    """
    _load_dotenv()
    backend = os.environ.get("BACKEND", "ollama").strip()
    council = [
        m.strip()
        for m in os.environ.get("COUNCIL_MODELS", "").split(",")
        if m.strip()
    ]
    judge = os.environ.get("JUDGE_MODEL", "").strip()

    if backend == "nim":
        base = os.environ.get(
            "NIM_BASE", "https://integrate.api.nvidia.com/v1"
        ).strip()
    else:
        base = os.environ.get("OLLAMA_BASE", "http://localhost:11434").strip()

    if not council or not judge:
        raise RuntimeError(
            "COUNCIL_MODELS and JUDGE_MODEL must be set. Run `make setup` first."
        )
    return council, judge, base


def _load_dotenv(path: str = ".env") -> None:
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())
