# Functional Specification

## Goal

An orchestration system where multiple small LLMs (the "council") answer a user question independently and in parallel, then a larger "judge" model deliberates and produces a single best final answer. Designed to run on hardware that cannot hold all models in VRAM at once.

Two surfaces are supported:
- **CLI** — `python council.py "..."` or `make run`.
- **Web** — FastAPI + Server-Sent Events on port 7860, served by `make web`.

## Core invariant: keep_alive=0

Every council generate request **must** set `keep_alive=0` in the Ollama payload. This forces immediate VRAM eviction when the response completes. Without it, models stay resident for 5 minutes (Ollama's default) and the sequential-loading scheme breaks.

Peak VRAM = `max(sum(council), judge)`, not `sum(council) + judge`.

Enforced by `tests/test_pipeline.py::test_keep_alive_zero_in_every_council_call`.

## Pipeline

1. User provides a prompt (CLI or web form).
2. Council models fire concurrently via `asyncio.gather`, each with `keep_alive=0`.
3. As each completes, its VRAM is released.
4. Judge loads (normal keep_alive) and runs two phases — see below.
5. Session is serialized to `results/session_<timestamp>.json`.

## Deliberative judge (two phases)

The judge is not a simple synthesizer — it is the senior reasoner.

**Phase 1 — Independent answer.**
Judge receives the user question with **no** council context and answers using only its own knowledge. The prompt instructs it to be concise, not hedge, and not reference other sources.

Between phases the judge stays hot via `keep_alive="5m"`, so Phase 2 incurs no load cost.

**Phase 2 — Review + final verdict.**
Judge receives: the original question, its own Phase 1 answer, and every council member's answer. It must:
1. For each council member, tag the answer as **CORRECT / PARTIALLY CORRECT / WRONG** with a one-line reason.
2. Call out any council detail that genuinely improves on its own answer.
3. Emit a single **FINAL VERDICT** — the authoritative answer the user sees.

Rationale: a naive "pick the best of three" prompt biases the judge toward the council even when the council is wrong. This flow anchors the judge on its own reasoning first.

The UI streams both phases into the same Synthesis panel, separated by divider headers ("▸ My own answer", "▸ Reviewing the council + final verdict").

## Model policy

`council/models.py` defines two lists:

- `APPROVED_MODELS` — production allowlist. Each entry records tag, role (council / judge / both), origin, and VRAM footprint in GB.
- `TEST_ONLY_MODELS` — permitted only for `make demo` and the test suite. Never valid in a production `.env`.

Approved vendors: Meta, Mistral AI, Google, Microsoft, Cohere. No Chinese-origin models in production.

Policy checks:
- `assert_production_allowed(tag, role)` rejects unknown or test-only models.
- `assert_council_diverse(tags)` rejects duplicate council members and councils of fewer than 2.
- Plans pick council members from **distinct origins**.
- Judge must be strictly larger than every council member.

## Setup wizard (`setup/setup.py`)

1. Ask local or remote.
2. Local path verifies Ollama is installed and running; starts it if needed. Remote path offers SSH tunnel or direct URL; SSH prints the manual command if the automatic launch fails.
3. Verify reachability via `GET /api/tags` before any pull.
4. Detect VRAM: `nvidia-smi` → `rocm-smi` → macOS `sysctl hw.memsize` → `/proc/meminfo`. If all fail, print per-OS manual commands and ask the user.
5. Build and render plans using `setup/plans.py`:
   - **Safe** — 2-model council, smaller judge, available at ~8 GB.
   - **Balanced** — shown only if VRAM ≥ 16 GB.
   - **Max** — shown only if VRAM ≥ 24 GB.
   - Peak VRAM = `max(sum(council), judge)`. Never render a plan that exceeds the budget.
6. Confirm download size, then pull via `POST /api/pull` with streaming progress.
7. Write `.env` (or `.env.demo` in demo mode).
8. After `make setup`: `make web` runs automatically.

## Web layer (`api/server.py` + `frontend/index.html`)

FastAPI app, served by `uvicorn` on port 7860.

**Routes:**
- `GET /` — serves the single-file HTML UI.
- `POST /ask` — `{"question": "..."}` → `{"session_id": "..."}`. Spawns a background asyncio task and returns immediately.
- `GET /stream/{sid}` — Server-Sent Events for a live session.
- `GET /history` — last 10 sessions from `results/`.
- `GET /health` — Ollama reachability + loaded models.

**SSE events** (emitted into the per-session asyncio queue):

| Event           | Payload                                                                 |
|-----------------|-------------------------------------------------------------------------|
| `council_start` | `{panel, model, origin, flag}`                                          |
| `council_chunk` | `{panel, text}`                                                         |
| `council_done`  | `{panel, latency, tokens}`                                              |
| `judge_start`   | `{model, origin, flag}`                                                 |
| `judge_chunk`   | `{text}` — both phases, with inline divider headers                     |
| `judge_done`    | `{latency, tokens, total_seconds, session_id, file}`                    |
| `error`         | `{message}`                                                             |

**Frontend contract:**
- Three grey panels ("Council Member 1/2/3") pulse amber during streaming, lock green on done.
- Synthesis panel slides up after all council panels complete.
- History sidebar hidden by default, toggled from footer. Clicking a history item replays council + judge panels inline.
- No flags or origin labels in panel heads (cosmetic decision).
- Copyright line: "© Vishaka Datta J Hebbar 2026".

The web layer calls `orch._stream_generate(...)` directly. `council/orchestrator.py` and `council/models.py` are **not modified** by the web layer.

## Demo mode (`make demo`)

- Skips remote setup and VRAM detection.
- Uses `TEST_ONLY_MODELS` only (default: `tinyllama` + `smollm2:1.7b` + `gemma2:2b` council, `llama3.2:3b` judge).
- Writes `.env.demo`; never touches `.env`.
- Pulls the tiny models if not present, then runs one hardcoded question.
- Prints `DEMO MODE — using tiny models, not suitable for real use.`

The orchestrator takes a `production=False` flag which disables the `APPROVED_MODELS` check, letting demo mode use tiny models without weakening the production policy.

## Results JSON

```json
{
  "timestamp": "2026-04-16T14:30:22",
  "prompt": "What is gravity?",
  "plan": "web",
  "council": [
    {
      "model": "tinyllama",
      "origin": "Unknown, —",
      "response": "...",
      "latency_seconds": 3.2,
      "tokens_generated": 87
    }
  ],
  "judge": {
    "model": "llama3.2:3b",
    "origin": "Meta, USA",
    "response": "▸ My own answer\n\n...\n\n▸ Reviewing the council + final verdict\n\n...",
    "latency_seconds": 8.4,
    "tokens_generated": 203
  },
  "total_seconds": 11.6
}
```

## Error handling

- Non-approved model in production → `ModelPolicyError` with a clear message.
- Duplicate council models → `ModelPolicyError`.
- Ollama unreachable at verify step → exit before any pull, print endpoint and HTTP error.
- Model pull error → surface Ollama's error line and abort setup.
- Runtime Ollama errors in the web layer → emitted to the client as an `error` SSE event, not swallowed.

## Testing

`tests/test_pipeline.py` uses `httpx.MockTransport` to fake the Ollama HTTP API. No GPU required. Covers:

- `keep_alive=0` invariant on every council call.
- Judge call does not use `keep_alive=0`.
- Production rejects duplicates, test-only, and unknown models.
- Production accepts approved models.
- Session JSON schema matches the documented shape.
- `production=False` bypasses the policy check.

## Out of scope

- The legacy Flask app under `legacy/` is not maintained.
- Function/tool calling — this is a text-in, text-out pipeline.
- Retrieval, memory, or multi-turn conversation — each request is independent.
