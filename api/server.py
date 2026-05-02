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
import statistics
import time
import uuid
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
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


class UltimateJudgeRequest(BaseModel):
    question: str
    council_summaries: list[dict] = []  # [{model, response}]
    previous_verdict: str = ""
    previous_judge_model: str = ""


ULTIMATE_JUDGE_MODEL = "meta/llama-3.1-405b-instruct"  # 405B — the boss


def _ultimate_judge_prompt(req: UltimateJudgeRequest) -> str:
    """
    The 405B is the court of appeal. We force a deliberate three-phase response:

      (1) reason from your own knowledge first — independent of the others
      (2) silently weigh that against the council + previous judge
      (3) deliver a final, definitive ruling — long enough to be authoritative,
          arrogant in tone, and never reduced to play-by-play "member X was right"

    The structure below is the prompt the 405B sees. The output structure
    (the headers) is what gets streamed back to the user.
    """
    parts = [
        "You are the ULTIMATE JUDGE — a 405-billion-parameter model.",
        "Smaller models have argued the question. A larger model has ruled. "
        "Some humans were unconvinced and summoned you. You are the final "
        "court of appeal. You do not negotiate. You do not hedge. You do "
        "not flatter the council or the previous judge. You issue the truth.",
        "",
        "Your output MUST follow this exact structure, in this order:",
        "",
        "  ### MY REASONING",
        "  Answer the original question yourself, from your own training, "
        "  ignoring everything below. Three to six sentences. Be substantive — "
        "  show actual reasoning, not just a one-line answer.",
        "",
        "  ### REVIEW",
        "  Now look at the council answers and the previous judge's verdict "
        "  below. ONE concise sentence per source: were they right, partially "
        "  right, or wrong, and on what specific point? Skip flattery. Only "
        "  call out a source if they got something materially wrong, or "
        "  surfaced a detail you missed.",
        "",
        "  ### FINAL VERDICT",
        "  Deliver the authoritative answer the user reads. Speak with "
        "  certainty. Do not say 'I think' or 'it appears'. Do not refer to "
        "  yourself as 'the Ultimate Judge' in this section — just speak the "
        "  truth directly. This section may run long if the question deserves "
        "  it. End with one sentence stating the answer in the cleanest form "
        "  possible.",
        "",
        "─── INPUTS ───",
        "",
        f"Original question:",
        f"  {req.question}",
        "",
        "Council answers:",
    ]
    for i, c in enumerate(req.council_summaries, 1):
        model = c.get("model", "?")
        body = (c.get("response") or "(no response)").strip()
        parts.append(f"  [Member {i} · {model}]")
        parts.append(f"    {body}")
        parts.append("")

    parts.append(f"Previous Judge ({req.previous_judge_model}) ruled:")
    parts.append(f"  {(req.previous_verdict or '(no verdict captured)').strip()}")
    parts.append("")
    parts.append("Begin now with `### MY REASONING`.")
    return "\n".join(parts)


@app.post("/ultimate-judge")
@limiter.limit("1/5minutes")
async def ultimate_judge(request: Request, req: UltimateJudgeRequest):
    """Summon the 405B — strictly rate-limited.
    Returns a session_id; subscribe to /stream/{sid} for the verdict."""
    if not req.question.strip():
        raise HTTPException(400, "empty question")

    client_id = _get_client_id(request)
    client_ip = (request.client.host if request.client else "unknown")

    sid: str = "ult-" + uuid.uuid4().hex[:12]
    queue: asyncio.Queue = asyncio.Queue()
    SESSIONS[sid] = queue

    asyncio.create_task(_run_ultimate_judge(sid, req, queue, client_id, client_ip))
    return {"session_id": sid, "model": ULTIMATE_JUDGE_MODEL}


async def _run_ultimate_judge(
    sid: str,
    req: UltimateJudgeRequest,
    queue: asyncio.Queue,
    client_id: str,
    client_ip: str,
) -> None:
    """Stream the 405B's verdict — no fallback, no auto-retry on 503.
    If the boss is busy, the user is told to come back later."""
    from council import nim_client

    await _emit(queue, "ultimate_start", {"model": ULTIMATE_JUDGE_MODEL})

    prompt = _ultimate_judge_prompt(req)
    chunks: list[str] = []
    tokens = 0
    t0 = time.perf_counter()

    server_model_seen: str | None = None
    try:
        # No fallback pool — defeats the whole "summon the boss" framing.
        # If 405B is queue-full, surface the error so the user can retry.
        async for text, raw in nim_client._stream_once(
            ULTIMATE_JUDGE_MODEL,
            prompt,
            os.environ.get("NVIDIA_API_KEY", ""),
            os.environ.get("NIM_BASE", "https://integrate.api.nvidia.com/v1").rstrip("/"),
            max_tokens=nim_client.ULTIMATE_MAX_TOKENS,
        ):
            if text:
                chunks.append(text)
                await _emit(queue, "ultimate_chunk", {"text": text})
            # Capture NIM's self-reported model — first chunk that has it
            if server_model_seen is None and raw.get("server_model"):
                server_model_seen = raw["server_model"]
                # Log to stdout — visible in Render logs as proof which model ran
                print(
                    f"[ULTIMATE-JUDGE] requested={ULTIMATE_JUDGE_MODEL} "
                    f"server={server_model_seen} client={client_id} ip={client_ip}",
                    flush=True,
                )
            if raw.get("done"):
                tokens = int(raw.get("eval_count", 0))
        latency = round(time.perf_counter() - t0, 2)

        await _emit(queue, "ultimate_done", {
            "latency":      latency,
            "tokens":       tokens,
            "model":        ULTIMATE_JUDGE_MODEL,
            "server_model": server_model_seen,  # NIM's own confirmation
        })
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 503:
            await _emit(queue, "error", {
                "message": "The Ultimate Judge is busy on NVIDIA's free tier. Try again in a minute."
            })
        else:
            await _emit(queue, "error", {"message": f"Ultimate Judge error: {e}"})
    except Exception as e:
        await _emit(queue, "error", {"message": f"{type(e).__name__}: {e}"})
    finally:
        await queue.put(None)


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
            judge_server_model: str | None = None
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
                # Capture NIM's self-reported judge model — proof of identity
                if judge_server_model is None and raw.get("server_model"):
                    judge_server_model = raw["server_model"]
                    print(
                        f"[JUDGE] requested={orch.judge_model} "
                        f"server={judge_server_model} client={client_id} ip={client_ip}",
                        flush=True,
                    )
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


# ---------------------------------------------------------------------------
# GPU Inference Observatory — /api/* routes
# Same service, same NVIDIA_API_KEY. Frontend at:
# https://vishakadatta.github.io/gpu-inference-infra/
# ---------------------------------------------------------------------------

_OBS_NIM_BASE      = "https://integrate.api.nvidia.com/v1"
_OBS_DEFAULT_MODEL = "meta/llama-3.1-8b-instruct"

_OBS_BLOCKED: frozenset[str] = frozenset({
    "deepseek-ai", "deepseek", "qwen", "alibaba", "01-ai",
    "thudm", "zhipu-ai", "zhipuai", "glm",
    "minimax-ai", "minimaxai", "moonshot-ai", "moonshotai",
    "baichuan-inc", "baichuan", "z-ai", "internlm", "openbmb",
})

_OBS_PROMPTS = {
    "short":  "What is a GPU?",
    "medium": (
        "Explain in detail how a graphics processing unit works, "
        "including its architecture, how it differs from a CPU, "
        "and why it is useful for machine learning workloads."
    ),
    "long": (
        "Write a comprehensive technical overview of GPU computing. "
        "Cover: the history of GPU development, CUDA programming model, "
        "GPU memory hierarchy (global, shared, registers, L1/L2 cache), "
        "thread execution model (warps, blocks, grids), common optimisation "
        "techniques, comparison with CPUs for parallel workloads, and the "
        "role of GPUs in modern AI inference and training."
    ),
}


def _obs_auth() -> str | None:
    key = os.environ.get("NVIDIA_API_KEY", "")
    return f"Bearer {key}" if key else None


def _obs_allowed(model_id: str) -> bool:
    if "/" not in model_id:
        return True
    return model_id.split("/")[0].lower() not in _OBS_BLOCKED


class _ObsInferReq(BaseModel):
    prompt:     str
    model:      Optional[str] = None
    max_tokens: int = 256


class _ObsLoadTestReq(BaseModel):
    prompt_preset: str = "short"
    concurrency:   int = 4
    num_requests:  int = 20
    max_tokens:    int = 100
    model:         Optional[str] = None


async def _obs_stream(prompt: str, model: str, max_tokens: int) -> dict:
    auth = _obs_auth()
    headers = {"Content-Type": "application/json"}
    if auth:
        headers["Authorization"] = auth
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
        "max_tokens": max_tokens,
        "temperature": 0.7,
    }
    start = time.perf_counter()
    ttft_ms = None
    full_text = ""
    token_count = 0
    prompt_tokens = 0
    model_used = model

    async with httpx.AsyncClient(timeout=120.0) as client:
        async with client.stream("POST", f"{_OBS_NIM_BASE}/chat/completions",
                                 json=payload, headers=headers) as resp:
            if resp.status_code != 200:
                body = await resp.aread()
                raise HTTPException(resp.status_code,
                                    f"NIM {resp.status_code}: {body.decode()[:300]}")
            async for raw_line in resp.aiter_lines():
                if not raw_line or not raw_line.startswith("data: "):
                    continue
                chunk = raw_line[6:]
                if chunk.strip() == "[DONE]":
                    break
                try:
                    obj = json.loads(chunk)
                except json.JSONDecodeError:
                    continue
                if obj.get("usage"):
                    prompt_tokens = obj["usage"].get("prompt_tokens", 0)
                choices = obj.get("choices", [])
                if not choices:
                    continue
                text = choices[0].get("delta", {}).get("content") or ""
                if text:
                    if ttft_ms is None:
                        ttft_ms = (time.perf_counter() - start) * 1000
                    full_text += text
                    token_count += 1
                if obj.get("model"):
                    model_used = obj["model"]

    total_ms = (time.perf_counter() - start) * 1000
    tps = token_count / (total_ms / 1000) if total_ms > 0 else 0.0
    return {
        "answer":            full_text,
        "ttft_ms":           round(ttft_ms or 0.0, 1),
        "total_latency_ms":  round(total_ms, 1),
        "tokens_generated":  token_count,
        "prompt_tokens":     prompt_tokens,
        "tokens_per_second": round(tps, 1),
        "model_used":        model_used,
        "backend":           "nim-hosted",
    }


async def _obs_single(prompt: str, model: str, max_tokens: int,
                      sem: asyncio.Semaphore, req_id: int) -> dict:
    auth = _obs_auth()
    headers = {"Content-Type": "application/json"}
    if auth:
        headers["Authorization"] = auth
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "max_tokens": max_tokens,
        "temperature": 0.7,
    }
    start = time.perf_counter()
    async with sem:
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                resp = await client.post(f"{_OBS_NIM_BASE}/chat/completions",
                                         json=payload, headers=headers)
            elapsed = (time.perf_counter() - start) * 1000
            if resp.status_code != 200:
                return {"req_id": req_id, "status": "error",
                        "latency_ms": round(elapsed, 2),
                        "error": f"HTTP {resp.status_code}"}
            tokens = resp.json().get("usage", {}).get("completion_tokens", 0)
            tps = tokens / (elapsed / 1000) if elapsed > 0 else 0.0
            return {"req_id": req_id, "status": "success",
                    "latency_ms": round(elapsed, 2),
                    "tokens_generated": tokens,
                    "tokens_per_second": round(tps, 2)}
        except Exception as e:
            elapsed = (time.perf_counter() - start) * 1000
            return {"req_id": req_id, "status": "error",
                    "latency_ms": round(elapsed, 2), "error": str(e)}


@app.get("/api/health")
async def obs_health():
    auth = _obs_auth()
    hdrs = {"Authorization": auth} if auth else {}
    start = time.perf_counter()
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.get(f"{_OBS_NIM_BASE}/models", headers=hdrs)
        ok = r.status_code == 200
        return {
            "status":      "ok" if ok else "degraded",
            "backend":     "nim-hosted",
            "model":       _OBS_DEFAULT_MODEL,
            "latency_ms":  round((time.perf_counter() - start) * 1000, 1),
            "http_status": r.status_code,
        }
    except Exception as e:
        return JSONResponse(status_code=503,
                            content={"status": "error", "detail": str(e)})


@app.get("/api/models")
async def obs_models():
    auth = _obs_auth()
    hdrs = {"Authorization": auth} if auth else {}
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            r = await client.get(f"{_OBS_NIM_BASE}/models", headers=hdrs)
        r.raise_for_status()
        models = [
            m["id"] for m in r.json().get("data", [])
            if _obs_allowed(m.get("id", ""))
        ]
        if _OBS_DEFAULT_MODEL in models:
            models.remove(_OBS_DEFAULT_MODEL)
            models.insert(0, _OBS_DEFAULT_MODEL)
        return {"models": models, "default": _OBS_DEFAULT_MODEL}
    except Exception as e:
        raise HTTPException(502, str(e))


@app.post("/api/infer")
@limiter.limit("10/minute")
async def obs_infer(request: Request, req: _ObsInferReq):
    model = req.model or _OBS_DEFAULT_MODEL
    try:
        return await _obs_stream(req.prompt, model, req.max_tokens)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(502, str(e))


@app.post("/api/loadtest")
@limiter.limit("10/minute")
async def obs_loadtest(request: Request, req: _ObsLoadTestReq):
    if req.prompt_preset not in _OBS_PROMPTS:
        raise HTTPException(400, f"prompt_preset must be one of {list(_OBS_PROMPTS)}")
    if req.concurrency > 16:
        raise HTTPException(400, "concurrency max is 16")
    if req.num_requests > 50:
        raise HTTPException(400, "num_requests max is 50")

    model  = req.model or _OBS_DEFAULT_MODEL
    prompt = _OBS_PROMPTS[req.prompt_preset]
    sem    = asyncio.Semaphore(req.concurrency)

    t0      = time.perf_counter()
    results = await asyncio.gather(*[
        _obs_single(prompt, model, req.max_tokens, sem, i)
        for i in range(req.num_requests)
    ])
    total_s = time.perf_counter() - t0

    successes = [r for r in results if r["status"] == "success"]
    errors    = [r for r in results if r["status"] == "error"]
    latencies = sorted(r["latency_ms"] for r in successes)
    n = len(latencies)

    def pct(p: float) -> float:
        if not latencies:
            return 0.0
        return round(latencies[min(int(p / 100 * n), n - 1)], 1)

    summary = {
        "total_requests":  req.num_requests,
        "successful":      len(successes),
        "errors":          len(errors),
        "error_rate_pct":  round(len(errors) / req.num_requests * 100, 1),
        "duration_s":      round(total_s, 2),
        "concurrency":     req.concurrency,
        "prompt_preset":   req.prompt_preset,
        "model":           model,
        "avg_latency_ms":  round(statistics.mean(latencies), 1) if latencies else 0,
        "p50_ms":          pct(50),
        "p95_ms":          pct(95),
        "p99_ms":          pct(99),
        "min_latency_ms":  round(latencies[0],  1) if latencies else 0,
        "max_latency_ms":  round(latencies[-1], 1) if latencies else 0,
        "avg_tps":         round(statistics.mean(
            r["tokens_per_second"] for r in successes), 1) if successes else 0,
    }
    return {"summary": summary, "results": results}
