# Functional Specification — Model Council

## Purpose

An orchestration system where multiple small LLMs (the "council") answer a question independently and in parallel, then a larger "judge" model deliberates and produces a single best final answer.

Two backends are supported:
- **NVIDIA NIM** — cloud-hosted models, no GPU needed, free tier available at [build.nvidia.com](https://build.nvidia.com)
- **Ollama** — fully local, GPU recommended

Two interfaces are supported:
- **Web UI** — FastAPI + Server-Sent Events on port 7860, live at https://vishakadatta.github.io/ModelCouncil/
- **CLI** — `python cli.py "..."` or `make run`

---

## Core pipeline

```
1. User submits a question (web form or CLI)
2. Council models run concurrently via asyncio.gather
   - NIM path: POST /v1/chat/completions with SSE streaming
   - Ollama path: POST /api/generate with keep_alive=0 (mandatory)
3. After all council members finish:
   Judge phase 1 — answers independently, no council context
   Judge phase 2 — reviews council answers vs its own, issues final verdict
4. Session saved to results/session_<timestamp>.json
```

---

## Ollama invariant: keep_alive=0

Every Ollama council generate request **must** set `keep_alive=0`. This forces immediate VRAM eviction when the response completes. Without it, models stay resident for 5 minutes (Ollama's default) and the system OOMs when the judge tries to load.

Peak VRAM = `max(sum(council), judge)` — not `sum(council) + judge`.

Enforced by `tests/test_pipeline.py::test_keep_alive_zero_in_every_council_call`. Do not remove this test or the constant.

The judge uses `keep_alive="5m"` to stay hot between its two phases so phase 2 incurs no reload cost.

This invariant is **Ollama-only**. The NIM backend never sends `keep_alive` — NVIDIA manages server-side memory.

---

## Deliberative judge

The judge is not a synthesizer — it is the senior reasoner. It runs in two sequential streaming phases:

**Phase 1 — Independent answer**

Prompt instructs the judge to answer using only its own knowledge. No council answers are visible. Output streams to the UI as "▸ My own answer".

**Phase 2 — Review and final verdict**

Prompt gives the judge: original question, its own phase 1 answer, and all council answers. It must:
1. For each council member: state CORRECT / PARTIALLY CORRECT / WRONG and why (one line each)
2. Note any council detail that genuinely improves on its own answer
3. Emit a single FINAL VERDICT — the authoritative answer the user reads

Output streams to the UI as "▸ Reviewing the council + final verdict".

Rationale: a naive "pick the best of three" prompt anchors the judge to the council before it thinks for itself, causing it to rubber-stamp wrong council answers. Phase 1 forces independent reasoning first.

---

## NIM backend

### Discovery (setup time)

`setup/nim_discover.py` queries `GET /v1/models` live and applies these rules in order:

| Rule | Action |
|------|--------|
| Publisher in `NIM_BLOCKED_PUBLISHERS` | Skip |
| Publisher not in `NIM_PUBLISHER_MAP` | Skip (only catalogued vendors are surfaced) |
| No parseable parameter count in model name | Skip |
| 15B ≤ params < 30B | Skip (ambiguous gap — too big for council, too small for judge) |
| params < 15B | council candidate |
| params ≥ 30B | judge candidate |
| Council has < 1 USA model | Reject entire set |
| Council has < 1 non-USA model | Accept if no non-USA available (warn) |

The `NIM_PUBLISHER_MAP` enumerates known vendors and their headquarters. The deny list (`NIM_BLOCKED_PUBLISHERS`) covers publishers whose origin metadata isn't easily verifiable from the catalogue alone, plus naming variants (e.g. `moonshotai` and `moonshot-ai` are both listed since NIM uses both forms).

### Runtime fallback

If a selected model returns 503 (queue full) or 404 (deprecated/not on free tier), `nim_client.stream_generate` automatically retries the next model from `COUNCIL_FALLBACK_POOL` or `JUDGE_FALLBACK_POOL`. The user never sees these errors.

Only 503 and 404 trigger a retry. Any other status code (401, 429, 500…) surfaces immediately.

### API details

- **Base URL:** `https://integrate.api.nvidia.com/v1`
- **Endpoint:** `POST /chat/completions`
- **Auth:** `Authorization: Bearer <NVIDIA_API_KEY>`
- **Streaming:** SSE (`data: {...}` lines, `data: [DONE]` terminator)
- **Model ID format:** `publisher/model-name` (e.g. `meta/llama-3.1-8b-instruct`)

---

## Ollama backend

### Discovery (setup time)

`setup/setup.py` detects VRAM and selects from `APPROVED_MODELS`:

| Plan | Council | Judge | Min VRAM |
|------|---------|-------|----------|
| Safe | 2 models (7–9B each) | 8B | ~8 GB |
| Balanced | 3 models (7–9B each) | 8B | ~16 GB |
| Max | 3 models (7–9B each) | 27B+ | ~24 GB |

Peak VRAM is computed as `max(sum(council), judge)` before presenting any plan. No plan is offered if it exceeds available VRAM.

### Model policy

`council/models.py` defines:

- `APPROVED_MODELS` — production allowlist. Tags: `llama3.1:8b`, `mistral:7b`, `gemma2:9b`, `phi3:mini`, `phi3:medium`, `command-r`, `gemma2:27b`, `command-r-plus`. Vendors: Meta, Mistral AI, Google, Microsoft, Cohere.
- `TEST_ONLY_MODELS` — `tinyllama`, `phi3:mini`, `gemma2:2b`, `qwen:0.5b`. Permitted only when `production=False`.

Policy checks:
- `assert_production_allowed(tag, role)` — rejects unknown or test-only models in production
- `assert_council_diverse(tags)` — rejects duplicates; requires ≥ 2 models

---

## Setup wizard

`setup/setup.py` — interactive wizard run via `make setup`:

1. **Choose backend:** Local Ollama or NVIDIA NIM
2. **NIM path:**
   - Prompt for API key (validated via `GET /v1/models`)
   - `nim_discover.discover_and_plan` queries live catalogue, applies policy, shows plan
   - User confirms or cancels
   - Writes `.env` with `BACKEND=nim`, `NVIDIA_API_KEY`, `NIM_BASE`, `COUNCIL_MODELS`, `JUDGE_MODEL`
3. **Ollama path:**
   - Verify Ollama installed and running (offers to install via brew/curl if not)
   - Detect VRAM: `nvidia-smi` → `rocm-smi` → macOS `sysctl hw.memsize` → `/proc/meminfo`
   - Build and display valid plans for detected VRAM
   - User selects plan, wizard pulls models, writes `.env` with `BACKEND=ollama`

---

## Web layer

### Server — `api/server.py`

FastAPI app served by `uvicorn` on port 7860 (local) or `$PORT` (Render).

**Config resolution — `_resolve_env()`:**
1. `BACKEND` already in `os.environ` → managed environment (Render, Docker) — use directly
2. `.env` file exists → local production
3. `.env.demo` exists → local demo
4. Nothing → `RuntimeError` with instructions

**Routes:**

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Serves `frontend/index.html` |
| `HEAD` | `/` | 200, no body (Render uptime probe) |
| `GET` | `/health` | Backend status — model list, connection check |
| `HEAD` | `/health` | 200, no body (proxy health checks) |
| `POST` | `/ask` | `{"question":"..."}` → `{"session_id":"..."}` |
| `GET` | `/stream/{sid}` | SSE event stream |
| `GET` | `/history` | Last 10 sessions from `results/` |

**CORS:** `allow_origins=["*"]` — required for GitHub Pages (different origin) to call Render API.

### SSE events

| Event | Payload | When |
|-------|---------|------|
| `council_start` | `{panel, model, origin, flag}` | Council member begins |
| `council_chunk` | `{panel, text}` | Token streamed |
| `council_done` | `{panel, latency, tokens}` | Council member finished |
| `judge_start` | `{model, origin, flag}` | Judge begins |
| `judge_chunk` | `{text}` | Judge token — both phases interleaved |
| `judge_done` | `{latency, tokens, total_seconds, session_id, file}` | Complete |
| `error` | `{message}` | Any exception |

### Frontend — `frontend/index.html`

Single static HTML file. No build step, no npm, no framework. Vanilla JS + `EventSource`.

- Three council panels pulse amber while streaming, lock green on `council_done`
- Judge "Synthesis" panel appears after all council panels complete
- Both judge phases stream into the same panel, separated by `▸` headers
- History sidebar hidden by default; toggle from footer; clicking a session replays its panels
- `window.COUNCIL_API_URL` — set to `http://localhost:7860` in the file; CI replaces it with the Render URL before deploying to GitHub Pages

---

## Demo mode

`make demo` → `make web`:
- `production=False` disables the `APPROVED_MODELS` check
- Uses `TEST_ONLY_MODELS` — smallest available models
- Writes `.env.demo`, never touches `.env`
- Suitable for CPU-only machines with no API key

---

## Session JSON

```json
{
  "timestamp": "2026-04-26T14:30:00",
  "prompt":    "What is gravity?",
  "plan":      "web",
  "council": [
    {
      "model":            "meta/llama-3.1-8b-instruct",
      "origin":           "Meta, USA",
      "response":         "Gravity is a fundamental force...",
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

The judge `response` always contains both phases. They can be split on `▸` for display or analysis.

---

## Error handling

| Scenario | Behaviour |
|----------|-----------|
| NIM 503 — queue full | Auto-retry next model in fallback pool |
| NIM 404 — model deprecated | Auto-retry next model in fallback pool |
| NIM 401 — bad API key | Surface immediately as SSE `error` event |
| All fallback models exhausted | `RuntimeError` surfaced as `error` event |
| Ollama unreachable at setup | Exit before any pull, print endpoint + HTTP error |
| Non-approved model in production | `ModelPolicyError` — caught at Orchestrator construction |
| Duplicate council models | `ModelPolicyError` — caught at Orchestrator construction |
| Empty question | FastAPI returns HTTP 400 |

---

## Testing

```bash
make test
# or: pytest tests/test_pipeline.py -v
```

14 tests. Uses `httpx.MockTransport` to fake both Ollama and NIM — no GPU, no network, no API key required. Runs in ~0.05s.

**Covered:**

| Test | Invariant |
|------|-----------|
| `test_keep_alive_zero_in_every_council_call` | `keep_alive=0` in every Ollama council call |
| `test_judge_call_does_not_use_keep_alive_zero` | Judge uses `"5m"`, not `0` |
| `test_council_diverse_rejects_duplicates` | Duplicate models raise `ModelPolicyError` |
| `test_test_only_model_rejected_in_production` | `tinyllama` blocked in production |
| `test_unknown_model_rejected` | Unknown tags rejected |
| `test_approved_model_accepted` | Known good tags pass |
| `test_full_session_writes_valid_json` | Session JSON matches schema |
| `test_orchestrator_rejects_duplicate_council_in_production` | Constructor enforces diversity |
| `test_orchestrator_allows_test_models_when_production_false` | Demo mode bypasses policy |
| `test_nim_blocks_chinese_publishers` | 20 blocked publishers verified |
| `test_nim_allows_western_publishers` | 6 western publishers pass |
| `test_nim_fetch_filters_chinese_models` | Mocked catalogue: Qwen + DeepSeek filtered |
| `test_nim_param_extraction` | 6 cases including MoE (8x7b → 56B) |
| `test_nim_role_classification` | Boundary values at 14.9B, 15.0B, 30.0B |

---

## Out of scope

- Function / tool calling — text in, text out only
- Multi-turn conversation — each request is independent
- Retrieval or memory — no RAG, no vector store
- Benchmarking / leaderboards — the output is one best answer, not a score table
