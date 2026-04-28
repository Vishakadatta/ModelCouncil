# Model Council — Functional & Technical Specification

> **Version:** 1.3
> **Last updated:** 2026-04-26
> **Status:** Live in production
> **Frontend:** https://vishakadatta.github.io/ModelCouncil/
> **Backend API:** https://llm-modelcouncil.onrender.com

---

## Changelog

| Version | Date | Summary |
|---------|------|---------|
| 1.0 | 2026-03-25 | Initial — Ollama backend, Flask UI, SQLite history |
| 1.1 | 2026-03-25 | Modular refactor, cross-judging, FastAPI + SSE rewrite |
| 1.2 | 2026-04-24 | NVIDIA NIM backend, dynamic model discovery, geographic diversity, Render + GitHub Pages deploy |
| 1.3 | 2026-04-26 | NIM fallback pool (503/404 auto-retry), expanded publisher policy, HEAD handlers, managed-env fix, model swap to gemma-3-4b-it |

---

## 1. What This Project Does

**Model Council** answers one question: *Can a group of small AI models, working together and judged by a larger model, match or beat that larger model working alone?*

The same question is sent to 3 small "council" models in parallel. A larger "judge" model then:
1. **Answers independently** (no council context)
2. **Reviews every council answer** — rates each CORRECT / PARTIALLY CORRECT / WRONG, notes anything better than its own answer
3. **Delivers a final verdict** — the single authoritative answer the user reads

This two-phase judge design (answer first, deliberate second) is the core differentiator. The judge can't be swayed by council answers before forming its own view.

### Access

| Where | URL | Notes |
|-------|-----|-------|
| Live website | https://vishakadatta.github.io/ModelCouncil/ | Anyone can use — no install |
| API | https://llm-modelcouncil.onrender.com | FastAPI, free Render tier |
| Local dev | `uvicorn api.server:app --port 7860` | Needs `.env` with API key |

---

## 2. Architecture

```
┌─────────────────────────────────────────────────┐
│  GitHub Pages (static)                          │
│  frontend/index.html                            │
│  window.COUNCIL_API_URL → Render backend URL    │
└───────────────────┬─────────────────────────────┘
                    │  HTTPS + SSE (EventSource)
                    ▼
┌─────────────────────────────────────────────────┐
│  Render (FastAPI)                               │
│  api/server.py                                  │
│  • POST /ask     → creates session, runs async  │
│  • GET /stream/{sid} → SSE event stream         │
│  • GET /health   → backend status               │
│  • GET /history  → last 10 sessions             │
└────────┬────────────────────┬───────────────────┘
         │                    │
         ▼                    ▼
┌────────────────┐   ┌────────────────────────┐
│ council/       │   │ results/               │
│ orchestrator   │   │ session_*.json         │
│ nim_client     │   │ (auto-saved per query) │
│ models         │   └────────────────────────┘
└────────┬───────┘
         │
         ▼
┌────────────────────────────────────────┐
│  NVIDIA NIM  (or Ollama, locally)      │
│  https://integrate.api.nvidia.com/v1   │
│  POST /chat/completions — SSE stream   │
└────────────────────────────────────────┘
```

---

## 3. Directory Structure

```
ModelCouncil/
├── api/
│   └── server.py           # FastAPI app — all HTTP endpoints + SSE session runner
│
├── council/
│   ├── __init__.py
│   ├── models.py           # Allowlists, NIM publisher map, blocked publishers, policy fns
│   ├── orchestrator.py     # Async council + judge pipeline; routes Ollama vs NIM
│   └── nim_client.py       # NIM streaming client; 503/404 fallback pool logic
│
├── setup/
│   ├── setup.py            # Interactive wizard: picks backend, discovers models, writes .env
│   └── nim_discover.py     # Live NIM discovery: filters, sizes, diversity, selects plan
│
├── frontend/
│   └── index.html          # Single-page UI — SSE streaming, council panels, judge panel
│
├── tests/
│   └── test_pipeline.py    # 14 tests — policy, keep_alive, session schema, NIM discovery
│
├── .github/workflows/
│   └── deploy.yml          # CI: inject Render URL → GitHub Pages; verify Render /health
│
├── render.yaml             # Render service definition + env var defaults
├── requirements.txt        # Python dependencies
├── Makefile                # make setup / make demo / make test / make run
└── SPEC.md                 # This file
```

---

## 4. Backends

### 4.1 NVIDIA NIM (production — what the live site uses)

**Base URL:** `https://integrate.api.nvidia.com/v1`
**Endpoint:** `POST /chat/completions` (OpenAI-compatible, SSE)
**Auth:** `Authorization: Bearer <NVIDIA_API_KEY>`

Models are specified by `publisher/model-name` IDs, e.g. `meta/llama-3.1-8b-instruct`.

#### Current model plan

| Role | Model | Origin | Size |
|------|-------|--------|------|
| Council 1 | `meta/llama-3.1-8b-instruct` | Meta, USA | 8B |
| Council 2 | `mistralai/mistral-7b-instruct-v0.3` | Mistral AI, France | 7B |
| Council 3 | `google/gemma-3-4b-it` | Google, USA | 4B — fast |
| Judge | `nvidia/llama-3.1-nemotron-70b-instruct` | NVIDIA, USA | 70B |

#### 503 / 404 fallback pool

NIM's free tier can queue-fill (503) or deprecate models (404). `nim_client.stream_generate` automatically retries the next model in the pool — the user never sees the error.

**Council pool** (13 models, tried in order on failure):
```
meta/llama-3.2-3b-instruct
google/gemma-3-4b-it
microsoft/phi-4-mini-instruct
meta/llama-3.2-1b-instruct
ibm/granite-3.0-3b-a800m-instruct
meta/llama-3.1-8b-instruct
mistralai/mistral-7b-instruct-v0.3
ibm/granite-3.0-8b-instruct
nvidia/llama-3.1-nemotron-nano-8b-v1
google/gemma-3-12b-it
nv-mistralai/mistral-nemo-12b-instruct
mistralai/ministral-14b-instruct-2512
microsoft/phi-3.5-moe-instruct
```

**Judge pool** (5 models):
```
nvidia/llama-3.1-nemotron-70b-instruct
meta/llama-3.1-70b-instruct
meta/llama-3.3-70b-instruct
mistralai/mistral-large
nvidia/llama-3.1-nemotron-51b-instruct
```

Only 503 (queue full) and 404 (model not on tier) trigger a retry. Any other status code (401, 429, 500…) surfaces immediately.

### 4.2 Ollama (local dev / no internet)

**Base URL:** `http://localhost:11434`
**Endpoint:** `POST /api/generate`

`keep_alive=0` is sent on **every** council call — non-negotiable. Forces immediate VRAM release so the judge can load. The judge uses `keep_alive="5m"` to stay hot between its two phases.

---

## 5. Module Reference

### 5.1 `council/models.py`

Central policy file. Nothing else in the codebase hardcodes model names or origins.

**Key symbols:**

| Symbol | Type | Purpose |
|--------|------|---------|
| `APPROVED_MODELS` | `list[ModelSpec]` | Ollama production allowlist |
| `TEST_ONLY_MODELS` | `list[str]` | Models allowed only in demo mode |
| `NIM_PUBLISHER_MAP` | `dict[str, (company, country)]` | Maps NIM publisher prefix → human label |
| `NIM_BLOCKED_PUBLISHERS` | `set[str]` | Chinese-origin publishers — always rejected |
| `origin_for(tag)` | `str` | Human-readable origin for any model tag (Ollama or NIM) |
| `assert_production_allowed(tag, role)` | raises `ModelPolicyError` | Ollama policy gate |
| `assert_council_diverse(tags)` | raises `ModelPolicyError` | No duplicate council models |

**Publisher allowlist approach:**

To keep the council deterministic and the dependency surface small, only publishers in `NIM_PUBLISHER_MAP` are eligible. New publishers can be added there as the project's vendor list expands. `NIM_BLOCKED_PUBLISHERS` is an explicit deny list for publishers we know we don't want to surface, primarily because their geographic-origin metadata isn't easily verifiable through NIM's catalogue alone.

> Implementation note: NIM uses no-hyphen publisher prefixes (`moonshotai`, `minimaxai`) — both hyphenated and un-hyphenated forms are listed in the deny set so naming variants don't slip through.

---

### 5.2 `council/nim_client.py`

Async NIM streaming client.

```python
async def stream_generate(
    model: str,
    prompt: str,
    api_key: str | None = None,
    base: str | None = None,
    fallback_pool: list[str] | None = None,
) -> AsyncIterator[tuple[str, dict]]:
```

Yields `(chunk_text, raw_dict)` — same interface as Ollama's path in `orchestrator._stream_generate`, so the pipeline needs zero changes when switching backends.

`raw_dict` keys:
```python
{
    "response":       str,   # token text (empty on final frame)
    "done":           bool,  # True only on the last frame
    "eval_count":     int,   # tokens generated (set when done=True)
    "model":          str,   # actual model used (may differ if fallback triggered)
    "fallback_model": str,   # present only on the first chunk of a fallback
}
```

**Retry logic:**
- Builds candidate list: `[requested_model] + [pool models not equal to requested]`
- On 503 or 404: logs, moves to next candidate
- On any other error: re-raises immediately
- All candidates exhausted: raises `RuntimeError` with full candidate list

---

### 5.3 `council/orchestrator.py`

Async council + judge pipeline.

**Key constant:**
```python
COUNCIL_KEEP_ALIVE = 0   # Ollama only — forces VRAM release after each council call
```

**`_stream_generate(client, model, prompt, keep_alive)`**

Routes to NIM or Ollama based on `BACKEND` env var read lazily at call time:

```python
backend = os.environ.get("BACKEND", "ollama")

if backend == "nim":
    pool = JUDGE_FALLBACK_POOL if model == self.judge_model else COUNCIL_FALLBACK_POOL
    async for chunk in nim_client.stream_generate(model, prompt, fallback_pool=pool):
        yield chunk
    return

# Ollama path — keep_alive=0 preserved for council, "5m" for judge
```

**`run_council(question)`** → `list[CouncilTurn]`

Fires all council models in parallel with `asyncio.gather`. Each call is non-streaming (Ollama path) for simplicity in tests. Returns structured turns.

**`run_judge_stream(question, council_turns)`** → `JudgeTurn`

Two sequential streaming calls to the judge:
1. Independent answer (no council context) — `keep_alive="5m"` to stay hot
2. Deliberation against council answers — produces final verdict

**`Session.to_dict()`** → JSON-serialisable dict matching the session schema documented in §7.

---

### 5.4 `api/server.py`

FastAPI application. All business logic lives in `orchestrator.py` — this file is wiring only.

**Endpoints:**

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Serves `frontend/index.html` |
| `HEAD` | `/` | 200 with no body (Render uptime probe) |
| `GET` | `/health` | Backend status JSON |
| `HEAD` | `/health` | 200 with no body (proxy health checks) |
| `POST` | `/ask` | Creates session, starts async pipeline, returns `{session_id}` |
| `GET` | `/stream/{sid}` | SSE event stream for a session |
| `GET` | `/history` | Last 10 saved sessions from `results/` |

**`_resolve_env()` — config priority:**
1. `BACKEND` already in `os.environ` → managed environment (Render, Docker) — use env vars directly, skip file loading
2. `.env` file exists → local production (`make setup`)
3. `.env.demo` file exists → local demo (`make demo`)
4. Nothing → raise `RuntimeError`

This means Render works with zero file changes — env vars injected by the dashboard are picked up automatically.

**SSE events emitted per session:**

| Event | Payload keys | When |
|-------|-------------|------|
| `council_start` | `panel, model, origin, flag` | Council member begins |
| `council_chunk` | `panel, text` | Token streamed |
| `council_done` | `panel, latency, tokens` | Council member finished |
| `judge_start` | `model, origin, flag` | Judge begins |
| `judge_chunk` | `text` | Judge token (both phases) |
| `judge_done` | `latency, tokens, total_seconds, session_id, file` | All done |
| `error` | `message` | Any exception |

---

### 5.5 `setup/nim_discover.py`

Live NIM model discovery — runs once at `make setup` time, writes `.env`.

**Policy rules applied in order:**

1. **Blocked origin** — publisher in `NIM_BLOCKED_PUBLISHERS` → skip
2. **Unknown origin** — publisher not in `NIM_PUBLISHER_MAP` → skip (unknown = not trusted)
3. **No parameter count** — name yields no parseable `Xb` → skip
4. **Role gap** — 15 ≤ params < 30B → skip (ambiguous tier)
5. **Geographic diversity** — council must have ≥1 USA and ≥1 non-USA model

**`_extract_param_b(model_id)`** — handles standard, fractional, sub-billion, and MoE patterns:
- `llama-3.1-8b` → 8.0
- `phi-3-mini-3.8b` → 3.8
- `mixtral-8x7b` → 56.0 (8×7)
- `llama-3.1-405b` → 405.0

**`_classify_role(param_b)`:**
- `< 15B` → `"council"`
- `≥ 30B` → `"judge"`
- `15–29B` → `None` (skipped — gap zone)

---

## 6. Session JSON Schema

Every completed query is written to `results/session_<timestamp>.json`.

```json
{
  "timestamp": "2026-04-26T14:30:00",
  "prompt": "What is gravity?",
  "plan": "web",
  "council": [
    {
      "model":            "meta/llama-3.1-8b-instruct",
      "origin":           "Meta, USA",
      "response":         "Gravity is...",
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

The judge `response` always contains both phases separated by the `▸` markers so they can be displayed or parsed independently.

---

## 7. Deployment

### GitHub Pages (frontend)

- **Source:** `frontend/index.html` — a single static HTML file, no build step
- **Deployed by:** `.github/workflows/deploy.yml` on every push to `main`
- **URL injection:** Before deploy, CI replaces:
  ```javascript
  // local fallback (in the file)
  window.COUNCIL_API_URL = window.COUNCIL_API_URL || "http://localhost:7860";
  // → injected for production
  window.COUNCIL_API_URL = "https://llm-modelcouncil.onrender.com";
  ```
- **Required GitHub secret:** `RENDER_BACKEND_URL = https://llm-modelcouncil.onrender.com`
- **Required Pages setting:** Source → **GitHub Actions** (not "Deploy from branch")

### Render (backend API)

- **Service type:** Web (Python)
- **Start command:** `uvicorn api.server:app --host 0.0.0.0 --port $PORT`
- **Health check path:** `/health`
- **Auto-deploy:** Yes — every push to `main` triggers rebuild

**Environment variables (set in Render dashboard):**

| Key | Value | Secret? |
|-----|-------|---------|
| `BACKEND` | `nim` | No |
| `NVIDIA_API_KEY` | `nvapi-...` | **Yes** |
| `NIM_BASE` | `https://integrate.api.nvidia.com/v1` | No |
| `COUNCIL_MODELS` | `meta/llama-3.1-8b-instruct,mistralai/mistral-7b-instruct-v0.3,google/gemma-3-4b-it` | No |
| `JUDGE_MODEL` | `nvidia/llama-3.1-nemotron-70b-instruct` | No |

> `render.yaml` provides defaults for non-secret vars. The dashboard value takes precedence — update the dashboard to change models without a redeploy.

### Local development

```bash
# 1. Copy env template and fill in your NIM key
cp .env.example .env
# edit NVIDIA_API_KEY in .env

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run
uvicorn api.server:app --port 7860 --reload

# 4. Open
open http://localhost:7860
```

Or use the interactive setup wizard:
```bash
make setup   # asks backend, discovers models, writes .env
make run     # starts the server
```

---

## 8. Tests

```bash
pytest tests/test_pipeline.py -v   # 14 tests, ~0.05s, no network, no GPU
```

**What's covered:**

| Test | What it verifies |
|------|-----------------|
| `test_council_diverse_rejects_duplicates` | Duplicate models in council raise `ModelPolicyError` |
| `test_test_only_model_rejected_in_production` | `tinyllama` blocked in production mode |
| `test_unknown_model_rejected` | Unknown model tags rejected |
| `test_approved_model_accepted` | Known good models pass |
| `test_keep_alive_zero_in_every_council_call` | `keep_alive=0` in every Ollama council call |
| `test_judge_call_does_not_use_keep_alive_zero` | Judge uses `"5m"` not `0` |
| `test_full_session_writes_valid_json` | Session JSON matches schema |
| `test_orchestrator_rejects_duplicate_council_in_production` | Constructor enforces diversity |
| `test_orchestrator_allows_test_models_when_production_false` | Demo mode bypasses policy |
| `test_nim_blocks_chinese_publishers` | 20 blocked publishers verified |
| `test_nim_allows_western_publishers` | 6 western publishers pass |
| `test_nim_fetch_filters_chinese_models` | Mocked catalogue: Qwen + DeepSeek filtered out |
| `test_nim_param_extraction` | 6 cases including MoE |
| `test_nim_role_classification` | Boundary values at 14.9B, 15.0B, 30.0B |

All tests use `FakeOllama` (mock transport) — no network, no GPU, no Ollama running required.

---

## 9. How to Contribute

### Swap a council model
1. Edit `COUNCIL_MODELS` in `.env` (local) or Render dashboard (production)
2. Restart server — no code change needed

### Add a publisher to the NIM map
Edit `NIM_PUBLISHER_MAP` in `council/models.py`:
```python
"newpublisher": ("Company Name", "Country"),
```

### Block a new publisher
Add to `NIM_BLOCKED_PUBLISHERS` in `council/models.py`. Include all known name variants (with and without hyphens).

### Add a fallback model
Append to `COUNCIL_FALLBACK_POOL` or `JUDGE_FALLBACK_POOL` in `council/nim_client.py`.

### Add a new SSE event
1. Define payload in `api/server.py` → `_emit(queue, "event_name", {...})`
2. Handle in `frontend/index.html` → `es.addEventListener("event_name", e => {...})`

### Run the full test suite
```bash
pytest tests/ -v
```

---

## 10. Known Limitations

| Limitation | Notes |
|-----------|-------|
| Render free tier cold starts | First request after inactivity takes ~30s. Render spins down idle services. |
| NIM free tier queue | Popular models (llama-3.1-8b) regularly hit 503. Fallback pool mitigates this. |
| No auth on the API | Anyone with the Render URL can make requests. NIM costs accrue to your key. |
| Session files accumulate | `results/` grows unbounded. Add cleanup if storage becomes a concern. |
| Judge bias | The judge forms its own answer before seeing council — but it's still a single model grading itself. |
| No streaming on Ollama council calls | `run_council` uses non-streaming generate. Only the judge streams on the Ollama path. |
