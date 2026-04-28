"""FastAPI + SSE web layer.

Wraps council.orchestrator.Orchestrator without modifying it.
Council model calls go through orchestrator._stream_generate which routes
to either Ollama (keep_alive=0 honoured) or NIM (keep_alive not sent).

DO NOT CHANGE: the judge two-phase logic (independent answer → deliberation).
"""

from __future__ import annotations

import asyncio
import json
import os
import time
import uuid
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from sse_starlette.sse import EventSourceResponse

from council.models import ModelPolicyError, origin_for
from council.orchestrator import (
    COUNCIL_KEEP_ALIVE,
    CouncilTurn,
    JudgeTurn,
    Orchestrator,
    Session,
    load_env_config,
)

ROOT        = Path(__file__).resolve().parent.parent
FRONTEND    = ROOT / "frontend" / "index.html"
RESULTS_DIR = ROOT / "results"
HISTORY_DB  = ROOT / "history.db"

# Admin token — set ADMIN_TOKEN in env to enable /admin/* routes.
# If unset, admin routes return 403.
ADMIN_TOKEN = os.environ.get("ADMIN_TOKEN", "")


# ---------------------------------------------------------------------------
# Admin SQLite store — every session, every client. Read-only for users.
# ---------------------------------------------------------------------------
import sqlite3

def _init_admin_db() -> None:
    """Create the sessions table if it doesn't exist. Idempotent."""
    with sqlite3.connect(HISTORY_DB) as db:
        db.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                client_id    TEXT    NOT NULL,
                timestamp    TEXT    NOT NULL,
                prompt       TEXT    NOT NULL,
                council_models TEXT,
                judge_model    TEXT,
                total_seconds  REAL,
                file_path    TEXT,
                ip           TEXT
            )
        """)
        db.execute("CREATE INDEX IF NOT EXISTS idx_client ON sessions(client_id)")
        db.execute("CREATE INDEX IF NOT EXISTS idx_ts ON sessions(timestamp DESC)")


def _record_session(
    client_id: str,
    session: "Session",
    file_path: str,
    ip: str,
) -> None:
    """Append a row to the admin DB. Best-effort — failures are logged not raised."""
    try:
        with sqlite3.connect(HISTORY_DB) as db:
            db.execute(
                """INSERT INTO sessions
                   (client_id, timestamp, prompt, council_models, judge_model,
                    total_seconds, file_path, ip)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    client_id,
                    session.timestamp,
                    session.prompt,
                    ",".join(t.model for t in session.council),
                    session.judge.model if session.judge else None,
                    session.total_seconds,
                    file_path,
                    ip,
                ),
            )
    except sqlite3.Error:
        pass  # best effort


_init_admin_db()

FLAGS = {
    "USA":    "🇺🇸",
    "France": "🇫🇷",
    "Canada": "🇨🇦",
    "UK":     "🇬🇧",
    "UAE":    "🇦🇪",
    "South Korea": "🇰🇷",
    "Japan":  "🇯🇵",
    "Israel": "🇮🇱",
}


def _flag_for(origin: str) -> str:
    for country, flag in FLAGS.items():
        if country in origin:
            return flag
    return "🏳️"


# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(title="Model Council")

# Rate limiter — keyed by client IP. Applied per-route via decorator.
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS — lock down to the GitHub Pages frontend (and localhost for dev).
# Anyone else calling this API from a browser will be rejected by CORS.
# Extra origins can be added via the ALLOWED_ORIGINS env var (comma-separated).
_default_origins = [
    "https://vishakadatta.github.io",
    "http://localhost:7860",
    "http://127.0.0.1:7860",
]
_extra = [
    o.strip()
    for o in os.environ.get("ALLOWED_ORIGINS", "").split(",")
    if o.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_default_origins + _extra,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# Per-session event queues; reader drains, producer pushes.
SESSIONS: dict[str, asyncio.Queue] = {}


def _resolve_env() -> tuple[list[str], str, str, bool]:
    """
    Resolve configuration from environment variables.

    Priority:
      1. Env vars already present in the process (Render, Docker, CI, etc.)
         If BACKEND is already set, skip file loading entirely — we're running
         in a managed environment where secrets are injected directly.
      2. .env file  (local production setup via `make setup`)
      3. .env.demo  (local demo mode via `make demo`)

    Returns (council_models, judge_model, base_url, is_production).
    """
    # 1. Managed environment (Render, etc.) — env vars already populated.
    if os.environ.get("BACKEND"):
        council, judge, base = load_env_config()
        return council, judge, base, True

    # 2–3. Local file-based setup.
    for path, production in ((".env", True), (".env.demo", False)):
        p = ROOT / path
        if not p.exists():
            continue
        for line in p.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())
        council, judge, base = load_env_config()
        return council, judge, base, production

    raise RuntimeError("No .env or .env.demo found. Run `make setup` or `make demo`.")


# ---------------------------------------------------------------------------
# REST endpoints
# ---------------------------------------------------------------------------

class AskRequest(BaseModel):
    question: str


@app.get("/")
async def index():
    if not FRONTEND.exists():
        raise HTTPException(500, "frontend/index.html missing")
    return FileResponse(FRONTEND)


@app.head("/")
async def index_head():
    """HEAD /  — Render's uptime probe hits this; return 200 with no body."""
    return Response(status_code=200)


@app.head("/health")
async def health_head():
    """HEAD /health — satisfy any HEAD-based health checks without running NIM auth."""
    return Response(status_code=200)


@app.get("/health")
async def health():
    try:
        council, judge, base, _ = _resolve_env()
    except RuntimeError as e:
        return {"status": "error", "backend": "unknown", "error": str(e)}

    backend = os.environ.get("BACKEND", "ollama")

    if backend == "nim":
        api_key = os.environ.get("NVIDIA_API_KEY", "")
        nim_ok  = False
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                r = await client.get(
                    "https://integrate.api.nvidia.com/v1/models",
                    headers={"Authorization": f"Bearer {api_key}"},
                )
                nim_ok = r.status_code == 200
        except httpx.HTTPError:
            nim_ok = False
        return {
            "status":  "ok" if nim_ok else "degraded",
            "backend": "nim",
            "nim":     "connected" if nim_ok else "unreachable",
            "models":  council + [judge],
            "council": council,
            "judge":   judge,
        }

    # Ollama backend
    ollama_ok = False
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            r = await client.get(f"{base}/api/tags")
            ollama_ok = r.status_code == 200
    except httpx.HTTPError:
        ollama_ok = False
    return {
        "status":  "ok" if ollama_ok else "degraded",
        "backend": "ollama",
        "ollama":  "connected" if ollama_ok else "unreachable",
        "base":    base,
        "models":  council + [judge],
        "council": council,
        "judge":   judge,
    }


def _get_client_id(request: Request) -> str:
    """Read X-Client-ID header. Returns 'anon' if missing — never blocks."""
    cid = request.headers.get("x-client-id", "").strip()
    return cid if cid else "anon"


@app.get("/history")
async def history(request: Request):
    """Return last 10 sessions for THIS client only.
    Filters by X-Client-ID header. 'anon' clients see nothing (privacy)."""
    client_id = _get_client_id(request)
    if client_id == "anon":
        return {"sessions": []}

    # Look up the file paths owned by this client_id from the admin DB.
    try:
        with sqlite3.connect(HISTORY_DB) as db:
            rows = db.execute(
                """SELECT file_path FROM sessions
                   WHERE client_id = ?
                   ORDER BY id DESC LIMIT 10""",
                (client_id,),
            ).fetchall()
    except sqlite3.Error:
        rows = []

    out = []
    for (fp,) in rows:
        if not fp:
            continue
        f = RESULTS_DIR / fp
        if not f.exists():
            continue
        try:
            data = json.loads(f.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        out.append({
            "file":          f.name,
            "timestamp":     data.get("timestamp"),
            "prompt":        data.get("prompt"),
            "plan":          data.get("plan"),
            "total_seconds": data.get("total_seconds"),
            "council_models": [c["model"] for c in data.get("council", [])],
            "judge_model":   (data.get("judge") or {}).get("model"),
            "data":          data,
        })
    return {"sessions": out}


@app.get("/admin/history")
async def admin_history(request: Request, token: str = ""):
    """Admin-only — see ALL sessions across all clients.
    Auth: pass ?token=... matching ADMIN_TOKEN env var."""
    if not ADMIN_TOKEN:
        raise HTTPException(403, "Admin route disabled (ADMIN_TOKEN not set)")
    if token != ADMIN_TOKEN:
        raise HTTPException(403, "Invalid admin token")

    try:
        with sqlite3.connect(HISTORY_DB) as db:
            db.row_factory = sqlite3.Row
            rows = db.execute(
                """SELECT id, client_id, timestamp, prompt, council_models,
                          judge_model, total_seconds, file_path, ip
                   FROM sessions
                   ORDER BY id DESC LIMIT 200"""
            ).fetchall()
    except sqlite3.Error as e:
        raise HTTPException(500, f"DB error: {e}")

    return {"sessions": [dict(r) for r in rows], "count": len(rows)}


@app.post("/ask")
@limiter.limit("10/minute")
async def ask(request: Request, req: AskRequest):
    question = req.question.strip()
    if not question:
        raise HTTPException(400, "empty question")

    client_id = _get_client_id(request)
    client_ip = (request.client.host if request.client else "unknown")

    council, judge, base, production = _resolve_env()

    try:
        orch = Orchestrator(
            council_models=council,
            judge_model=judge,
            ollama_base=base,
            plan_name="web",
            production=production,
        )
    except ModelPolicyError as e:
        raise HTTPException(400, str(e))

    sid: str = uuid.uuid4().hex[:12]
    queue: asyncio.Queue = asyncio.Queue()
    SESSIONS[sid] = queue
    asyncio.create_task(_run_session(sid, question, orch, queue, client_id, client_ip))
    return {"session_id": sid}


@app.get("/stream/{sid}")
async def stream(sid: str):
    queue = SESSIONS.get(sid)
    if queue is None:
        raise HTTPException(404, "unknown session")

    async def event_gen():
        try:
            while True:
                evt = await queue.get()
                if evt is None:
                    break
                yield evt
        finally:
            SESSIONS.pop(sid, None)

    return EventSourceResponse(event_gen())


# ---------------------------------------------------------------------------
# Session runner — emits SSE events into the session queue
# DO NOT CHANGE: the judge two-phase logic below.
# ---------------------------------------------------------------------------

async def _emit(queue: asyncio.Queue, event: str, payload: dict) -> None:
    await queue.put({"event": event, "data": json.dumps(payload)})


def _judge_independent_prompt(question: str) -> str:
    """Step 1: judge answers on its own with no council context."""
    return (
        "You are a careful senior reasoner. Answer the following question "
        "using only your own knowledge and reasoning. Be concise but complete. "
        "Do not hedge. Do not reference any other sources or assistants.\n\n"
        f"Question: {question}\n\n"
        "Your answer:"
    )


def _judge_deliberation_prompt(
    question: str,
    own_answer: str,
    council_turns: list[CouncilTurn],
) -> str:
    """Step 2: judge weighs its own answer against each council member's."""
    parts = [
        "You are the judge. You have already produced your own independent "
        "answer to a user question. Three smaller council models have also "
        "answered. You are a larger model with stronger reasoning, so trust "
        "your own answer as the baseline — but check whether any council "
        "member surfaced a detail you missed or a correction worth adopting.",
        "",
        "Do this in order:",
        "  1. For each council member, in one short line, say whether their "
        "answer is CORRECT, PARTIALLY CORRECT, or WRONG — and why.",
        "  2. Note any specific point from a council answer that improves on "
        "yours, if any.",
        "  3. Give the FINAL VERDICT — a single clean, authoritative answer "
        "to the original question. This is what the user will read.",
        "",
        f"Original question: {question}",
        "",
        "Your own answer (from Step 1):",
        own_answer,
        "",
        "Council answers:",
    ]
    for i, t in enumerate(council_turns, 1):
        parts.append(f"  Council member {i} ({t.model}): {t.response}")
    parts.append("")
    parts.append("Begin with step 1.")
    return "\n".join(parts)


async def _stream_one_council(
    orch: Orchestrator,
    client: httpx.AsyncClient,
    panel: int,
    model: str,
    question: str,
    queue: asyncio.Queue,
) -> CouncilTurn:
    origin = origin_for(model)
    await _emit(queue, "council_start", {
        "panel":  panel,
        "model":  model,
        "origin": origin,
        "flag":   _flag_for(origin),
    })

    chunks: list[str] = []
    tokens = 0
    t0 = time.perf_counter()
    async for text, raw in orch._stream_generate(
        client, model, question, keep_alive=COUNCIL_KEEP_ALIVE
    ):
        if text:
            chunks.append(text)
            await _emit(queue, "council_chunk", {"panel": panel, "text": text})
        if raw.get("done"):
            tokens = int(raw.get("eval_count", 0))
    latency = round(time.perf_counter() - t0, 2)

    await _emit(queue, "council_done", {
        "panel":   panel,
        "latency": latency,
        "tokens":  tokens,
    })

    return CouncilTurn(
        model=model,
        origin=origin,
        response="".join(chunks).strip(),
        latency_seconds=latency,
        tokens_generated=tokens,
    )


async def _run_session(
    sid: str,
    question: str,
    orch: Orchestrator,
    queue: asyncio.Queue,
    client_id: str = "anon",
    client_ip: str = "unknown",
) -> None:
    try:
        async with httpx.AsyncClient(timeout=600.0) as client:
            # ── Council (all members in parallel) ─────────────────────────
            council_turns = await asyncio.gather(*[
                _stream_one_council(orch, client, i, model, question, queue)
                for i, model in enumerate(orch.council_models)
            ])

            judge_origin = origin_for(orch.judge_model)
            await _emit(queue, "judge_start", {
                "model":  orch.judge_model,
                "origin": judge_origin,
                "flag":   _flag_for(judge_origin),
            })

            # ── Judge phase 1: independent answer ─────────────────────────
            own_chunks: list[str] = []
            own_tokens = 0
            t0 = time.perf_counter()
            await _emit(queue, "judge_chunk", {"text": "▸ My own answer\n\n"})
            async for text, raw in orch._stream_generate(
                client,
                orch.judge_model,
                _judge_independent_prompt(question),
                keep_alive="5m",  # stay hot for phase 2
            ):
                if text:
                    own_chunks.append(text)
                    await _emit(queue, "judge_chunk", {"text": text})
                if raw.get("done"):
                    own_tokens = int(raw.get("eval_count", 0))
            own_answer = "".join(own_chunks).strip()

            # ── Judge phase 2: deliberation against council ────────────────
            await _emit(queue, "judge_chunk", {
                "text": "\n\n▸ Reviewing the council + final verdict\n\n"
            })
            deliberation_prompt = _judge_deliberation_prompt(
                question, own_answer, council_turns
            )
            delib_chunks: list[str] = []
            delib_tokens = 0
            async for text, raw in orch._stream_generate(
                client, orch.judge_model, deliberation_prompt, keep_alive="5m"
            ):
                if text:
                    delib_chunks.append(text)
                    await _emit(queue, "judge_chunk", {"text": text})
                if raw.get("done"):
                    delib_tokens = int(raw.get("eval_count", 0))
            judge_latency = round(time.perf_counter() - t0, 2)

            judge_turn = JudgeTurn(
                model=orch.judge_model,
                origin=judge_origin,
                response=(
                    "▸ My own answer\n\n"
                    + own_answer
                    + "\n\n▸ Reviewing the council + final verdict\n\n"
                    + "".join(delib_chunks).strip()
                ),
                latency_seconds=judge_latency,
                tokens_generated=own_tokens + delib_tokens,
            )

        session = Session(
            timestamp=datetime.now().isoformat(timespec="seconds"),
            prompt=question,
            plan=orch.plan_name,
            council=council_turns,
            judge=judge_turn,
            total_seconds=round(
                sum(t.latency_seconds for t in council_turns) + judge_latency, 2
            ),
        )
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out   = RESULTS_DIR / f"session_{stamp}.json"
        out.write_text(json.dumps(session.to_dict(), indent=2))

        # Record in admin DB so /history (per-client) and /admin/history (all)
        # can find it. File on disk is the source of truth; DB is the index.
        _record_session(client_id, session, out.name, client_ip)

        await _emit(queue, "judge_done", {
            "latency":       judge_latency,
            "tokens":        own_tokens + delib_tokens,
            "total_seconds": session.total_seconds,
            "session_id":    sid,
            "file":          out.name,
        })

    except httpx.HTTPError as e:
        await _emit(queue, "error", {"message": f"Backend error: {e}"})
    except Exception as e:
        await _emit(queue, "error", {"message": f"{type(e).__name__}: {e}"})
    finally:
        await queue.put(None)
