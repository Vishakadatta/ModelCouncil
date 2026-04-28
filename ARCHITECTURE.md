# Architecture — Model Council

A self-contained technical overview. Start here if you want to understand how the system fits together before reading code.

---

## What it is

Model Council is an LLM orchestration pipeline with two backends (NVIDIA NIM cloud, Ollama local) and two interfaces (web UI, CLI). The same question goes to 3 small "council" models in parallel, then a larger "judge" model deliberates and issues a final verdict.

It is deployed as:
- A **static frontend** on GitHub Pages (anyone can use it)
- A **FastAPI backend** on Render (free tier, auto-deploys from `main`)

It can also run fully locally with `make setup`.

---

## Deployment topology

```
┌──────────────────────────────────────┐
│  GitHub Pages                        │
│  frontend/index.html (static)        │
│                                      │
│  window.COUNCIL_API_URL              │
│  = "https://llm-modelcouncil..."     │  ← injected by CI before deploy
│    OR "http://localhost:7860"        │  ← fallback for local dev
└────────────────┬─────────────────────┘
                 │ HTTPS + SSE
                 ▼
┌──────────────────────────────────────┐
│  Render (FastAPI)                    │
│  api/server.py                       │
│  uvicorn on $PORT                    │
│  BACKEND=nim                         │
│  NVIDIA_API_KEY=...                  │
└────────────────┬─────────────────────┘
                 │
                 ▼
┌──────────────────────────────────────┐
│  NVIDIA NIM                          │
│  integrate.api.nvidia.com/v1         │
│  POST /chat/completions (SSE)        │
│  OpenAI-compatible wire format       │
└──────────────────────────────────────┘
```

For local dev, Render is replaced by `uvicorn api.server:app --port 7860` on your machine, and NIM can be swapped for a local Ollama daemon.

---

## Request lifecycle

```
1. Browser POSTs {"question": "..."} to /ask
2. FastAPI creates a UUID session, starts a background asyncio task, returns {session_id}
3. Browser opens GET /stream/{sid} — SSE connection
4. Background task:
     a. Fire council_models in parallel via asyncio.gather
        Each council call → NIM /chat/completions (streaming)
        Emit council_start, council_chunk×N, council_done per model
     b. After all 3 finish:
        Judge phase 1 → NIM /chat/completions (streaming)
        Emit judge_start, judge_chunk×N (phase 1 text)
        Judge phase 2 → NIM /chat/completions (streaming)
        Emit judge_chunk×N (phase 2 text), judge_done
     c. Write results/session_<timestamp>.json
     d. Push None sentinel → SSE connection closes
```

---

## Backend routing

`BACKEND` env var selects the transport at runtime. `council/orchestrator.py` reads it lazily in `_stream_generate` so the same code handles both paths:

```
BACKEND=nim    → council/nim_client.py  → NIM /v1/chat/completions
BACKEND=ollama → inline httpx call      → Ollama /api/generate
```

**Ollama-specific:** `keep_alive=0` is sent on every council call. This forces immediate VRAM eviction the moment a response completes, so the judge can load without OOM. The judge uses `keep_alive="5m"` to stay hot between its two phases. This invariant is enforced by the test suite.

**NIM-specific:** `keep_alive` is never sent — it's an Ollama concept. NIM manages memory server-side.

---

## NIM client and fallback pool

`council/nim_client.py` wraps `POST /v1/chat/completions` into the same `(text, raw_dict)` async generator interface Ollama uses — so `orchestrator._stream_generate` needs no branching above the transport level.

NVIDIA's free tier returns **503** when queues fill and **404** when a model is deprecated. Both are retried automatically from a fallback pool:

- **Council pool:** 13 small western text models (<30B)
- **Judge pool:** 5 large models (≥50B)

The first model in `COUNCIL_MODELS` is tried first; on 503/404 the client silently moves to the next pool entry. Any other status code (401, 429, 500…) surfaces immediately.

---

## Model policy

`council/models.py` is the single source of truth for what models are allowed.

### NIM backend

Models are discovered live at setup time by `setup/nim_discover.py`. Rules applied in order:

1. **Deny list** — publishers in `NIM_BLOCKED_PUBLISHERS` are skipped
2. **Unknown publisher** — not in `NIM_PUBLISHER_MAP` → skipped (only catalogued vendors are surfaced)
3. **No size** — can't parse a parameter count → skipped
4. **Gap zone** — 15–29B models → skipped (too big for council, too small for judge)
5. **Geographic diversity** — council must mix US and non-US labs

The `NIM_PUBLISHER_MAP` is the source of truth — adding a publisher is a single-line change in `council/models.py`.

### Ollama backend

Static `APPROVED_MODELS` allowlist (Meta, Mistral AI, Google, Microsoft, Cohere). `assert_production_allowed` enforces it. `TEST_ONLY_MODELS` (tiny models) are permitted only when `production=False`.

---

## Deliberative judge design

The judge runs in two phases to prevent anchoring bias:

```
Phase 1 — Independent answer
  Prompt: "Answer this question using only your own knowledge."
  Judge sees: question only
  Output: judge's own answer (streamed to UI)
  keep_alive: "5m" (stays hot for phase 2)

Phase 2 — Review and verdict
  Prompt: "You answered X. The council answered A, B, C.
           For each, say CORRECT/WRONG/PARTIAL and why.
           Then give the FINAL VERDICT."
  Judge sees: its own phase 1 answer + all 3 council answers
  Output: per-model ratings + single authoritative verdict
```

Without phase 1, a naive "pick the best of three" judge has been observed to agree with the council even when the council is wrong. Phase 1 anchors the judge on its own reasoning first.

Both phases stream into the same "Synthesis" panel in the UI, separated by `▸` dividers.

---

## SSE protocol

```
POST /ask → {session_id}
GET /stream/{sid} → text/event-stream

Events:

council_start  {panel, model, origin, flag}
council_chunk  {panel, text}
council_done   {panel, latency, tokens}
judge_start    {model, origin, flag}
judge_chunk    {text}                        ← both phases, same stream
judge_done     {latency, tokens, total_seconds, session_id, file}
error          {message}
```

Each session has its own `asyncio.Queue`. The background task pushes events; the SSE generator drains it. A `None` sentinel closes the stream.

---

## Session JSON schema

Written to `results/session_<timestamp>.json` after every query:

```json
{
  "timestamp": "2026-04-26T14:30:00",
  "prompt":    "What is gravity?",
  "plan":      "web",
  "council": [
    {
      "model":            "meta/llama-3.1-8b-instruct",
      "origin":           "Meta, USA",
      "response":         "...",
      "latency_seconds":  3.2,
      "tokens_generated": 142
    }
  ],
  "judge": {
    "model":            "nvidia/llama-3.1-nemotron-70b-instruct",
    "origin":           "NVIDIA, USA",
    "response":         "▸ My own answer\n\n...\n\n▸ Reviewing the council + final verdict\n\n...",
    "latency_seconds":  8.1,
    "tokens_generated": 389
  },
  "total_seconds": 11.3
}
```

No database. History is a directory of JSON files sorted by mtime. The `/history` endpoint returns the last 10.

---

## CI/CD

`.github/workflows/deploy.yml` runs on every push to `main`:

1. **deploy-frontend** — Python script replaces the `localhost:7860` fallback URL in `index.html` with the `RENDER_BACKEND_URL` secret, then uploads `frontend/` as a GitHub Pages artifact and deploys.
2. **verify-backend** — hits `GET /health` on the Render URL; emits a warning (not failure) if the backend is cold-starting.

Render auto-deploys from the same push via its GitHub integration.

---

## Key invariants and where they are enforced

| Invariant | Enforced by |
|-----------|-------------|
| `keep_alive=0` on every Ollama council call | `test_keep_alive_zero_in_every_council_call` |
| Judge does not use `keep_alive=0` | `test_judge_call_does_not_use_keep_alive_zero` |
| Council models are distinct | `assert_council_diverse` + Orchestrator constructor |
| Production only uses approved models | `assert_production_allowed` + `ModelPolicyError` |
| Test-only models never in production | Separate `TEST_ONLY_MODELS` list + `production` flag |
| Chinese-origin models never selected | `NIM_BLOCKED_PUBLISHERS` checked in `fetch_nim_models` |
| 503/404 never surface to user | `nim_client.stream_generate` fallback loop |

---

© Vishaka Datta J Hebbar 2026
