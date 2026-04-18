# Model Council

A Python orchestration system where several smaller "council" LLMs answer a question in parallel over Ollama, then a larger "judge" model synthesizes a single best final answer. Each council model is evicted from VRAM the moment it finishes, so the judge can load on hardware that could never hold every model simultaneously.

The judge is **deliberative**: it first answers the question on its own, then reviews the council's answers against its own reasoning before issuing a final verdict.

---

## Architecture

```
                          User prompt
                               │
                               ▼
  ┌──────────────────────────────────────────────────┐
  │  Council — parallel via asyncio.gather           │
  │                                                  │
  │   Member 1          Member 2         Member 3    │
  │  (e.g. tinyllama)  (smollm2:1.7b)  (gemma2:2b)   │
  │       │                │                │        │
  │  keep_alive=0     keep_alive=0    keep_alive=0   │
  │       │                │                │        │
  │       ▼                ▼                ▼        │
  │        VRAM purged after each answer             │
  └──────────────────────────────────────────────────┘
                               │
                               ▼
                    VRAM is now empty
                               │
                               ▼
  ┌──────────────────────────────────────────────────┐
  │  Judge — loads alone (e.g. llama3.2:3b)          │
  │                                                  │
  │  Phase 1: answer independently, no council ctx   │
  │            keep_alive="5m" (stay hot)            │
  │                       │                          │
  │                       ▼                          │
  │  Phase 2: review each council answer vs own,     │
  │           then emit FINAL VERDICT                │
  └──────────────────────────────────────────────────┘
                               │
                               ▼
              results/session_TIMESTAMP.json
```

### Why `keep_alive=0` is the key

Ollama's `keep_alive` parameter controls how long a model stays resident in VRAM after a request. Default is 5 minutes — fine for chat, fatal for sequential loading. Setting `keep_alive=0` on every council request forces immediate eviction the instant the response completes.

Peak VRAM is therefore `max(sum(council), judge)`, **not** `sum(council) + judge`. This invariant is asserted by the test suite (`tests/test_pipeline.py::test_keep_alive_zero_in_every_council_call`).

### Why the judge deliberates in two phases

A naive judge prompt ("here are three answers, pick the best") biases the judge toward the council even when the council is wrong. The two-phase flow defends against this:

1. **Independent answer** — judge answers the user's question with only its own knowledge. No council output visible.
2. **Review + verdict** — judge compares each council answer against its own, tags each as CORRECT / PARTIALLY CORRECT / WRONG, absorbs any real improvements, and emits a single authoritative final answer.

Between the two phases the judge stays resident (`keep_alive="5m"`) so there is no second load cost.

---

## Quick start

```bash
git clone <this repo>
cd ModelCouncil
pip install -r requirements.txt
make setup          # interactive wizard → auto-launches the web UI on :7860
```

Or try the pipeline end-to-end with tiny CPU-friendly models first:

```bash
make demo           # writes .env.demo, pulls tiny models, runs one CLI question
make web            # start the web UI on http://localhost:7860
python council.py "What is gravity?"   # pure-CLI path, no browser
```

---

## Web GUI

`make web` starts a FastAPI server on **port 7860** and opens your browser.

- **Backend** — [api/server.py](api/server.py). Thin FastAPI layer that consumes `Orchestrator._stream_generate` directly so every council model streams token-by-token over Server-Sent Events. `council/orchestrator.py` is untouched by the web layer.
- **Frontend** — [frontend/index.html](frontend/index.html). Single file, vanilla JS, no build step, no npm. Dark editorial aesthetic (`#0a0a0a` / `#111` / `#f5a623`) with IBM Plex Mono for headers and Inter for body.
- **History sidebar** — hidden by default; a toggle button in the footer reveals the last 10 sessions. Clicking a session replays its council + judge panels inline.
- **Protocol** — SSE events described below.

### SSE event protocol

| Event           | Payload                                                                    |
|-----------------|----------------------------------------------------------------------------|
| `council_start` | `{panel, model, origin, flag}`                                             |
| `council_chunk` | `{panel, text}` — streamed token delta                                     |
| `council_done`  | `{panel, latency, tokens}`                                                 |
| `judge_start`   | `{model, origin, flag}`                                                    |
| `judge_chunk`   | `{text}` — interleaves Phase 1 answer, divider, and Phase 2 review+verdict |
| `judge_done`    | `{latency, tokens, total_seconds, session_id, file}`                       |
| `error`         | `{message}`                                                                |

### REST routes

| Method | Path             | Purpose                                                               |
|--------|------------------|-----------------------------------------------------------------------|
| `GET`  | `/`              | Serves `frontend/index.html`                                          |
| `POST` | `/ask`           | `{"question": "..."}` → `{"session_id": "..."}`; spawns background run |
| `GET`  | `/stream/{sid}`  | Server-Sent Events for a live session                                 |
| `GET`  | `/history`       | Last 10 sessions from `results/`                                      |
| `GET`  | `/health`        | Ollama reachability + loaded models                                   |

The UI keeps each council panel inert grey until its model starts, pulses amber while streaming, and locks to green with latency on completion. The judge "Synthesis" panel slides up after all three council members finish.

---

## Hardware tiers

| VRAM    | Plans available        | Example plan                                           |
|---------|------------------------|--------------------------------------------------------|
| ~4 GB   | Demo only              | Council: tinyllama + smollm2:1.7b + gemma2:2b · Judge: llama3.2:3b |
| 8 GB    | Safe                   | Council: phi3:mini + mistral:7b · Judge: llama3.1:8b   |
| 16 GB   | Safe + Balanced        | Council: 3× 7–9B models · Judge: llama3.1:8b           |
| 24 GB   | Safe + Balanced + Max  | Council: 3× diverse 7–9B · Judge: gemma2:27b           |
| 32 GB+  | All three              | Max plan with command-r-plus judge                     |

The wizard never offers a plan whose peak VRAM exceeds what you have available.

---

## How to run (for users)

**Full setup (recommended):**
```bash
make setup
```
Interactive wizard walks you through local vs. remote Ollama, VRAM detection, plan selection, and model pulls. When it finishes, your browser opens on the web UI automatically.

**Just the web UI (if `.env` already exists):**
```bash
make web
```

**CPU-only demo (no GPU required):**
```bash
make demo
```
Writes a separate `.env.demo` with `tinyllama` + `smollm2:1.7b` + `gemma2:2b` council and `llama3.2:3b` judge. Runs a single hardcoded question through the CLI path to prove the pipeline end-to-end.

**Pure CLI (no browser):**
```bash
python council.py "Your question here"
```

**Results:**
```bash
make results   # pretty-prints the latest session JSON
```

**Tests (no GPU needed; uses httpx.MockTransport):**
```bash
make test
```

---

## Model policy

This project uses a **hardcoded allowlist**. Only these models can appear in a production plan:

| Tag              | Role          | Origin               |
|------------------|---------------|----------------------|
| llama3.1:8b      | council/judge | Meta, USA            |
| mistral:7b       | council/judge | Mistral AI, France   |
| gemma2:9b        | council       | Google, USA          |
| gemma2:27b       | judge         | Google, USA          |
| phi3:mini        | council       | Microsoft, USA       |
| phi3:medium      | council       | Microsoft, USA       |
| command-r        | council       | Cohere, Canada       |
| command-r-plus   | judge         | Cohere, Canada       |

Approved vendors: Meta, Mistral AI, Google, Microsoft, Cohere. **No Chinese-origin models** (Qwen, DeepSeek, Yi, etc.) are permitted in production plans — this project runs in contexts with supply-chain and data-provenance constraints that require a vetted vendor list. Adding a vendor requires editing [council/models.py](council/models.py) and is a deliberate policy change.

A separate `TEST_ONLY_MODELS` bucket (`tinyllama`, `smollm2:1.7b`, `phi3:mini`, `gemma2:2b`, `llama3.2:3b`, `qwen:0.5b`) exists exclusively for `make demo` and the test suite. The orchestrator raises `ModelPolicyError` if any of them appear in a production `.env`.

Council members must come from **distinct vendors** — no two models from the same origin in the same run. The judge must be strictly larger than every council member.

---

## Remote GPU

The setup wizard supports two remote modes:

- **SSH tunnel** — wizard prompts for host/user/port and runs `ssh -L 11434:localhost:11434 USER@HOST -p PORT -N &`. If the automatic launch fails the exact command is printed so you can run it manually, and the wizard waits for you to confirm before continuing.
- **Direct URL** — point `OLLAMA_BASE` at an existing reachable Ollama endpoint.

Both modes verify connectivity via `GET /api/tags` **before** any model is pulled, so you never discover a broken connection mid-download.

---

## Testing

```bash
make test
```

Tests use `httpx.MockTransport` to fake Ollama, so they run on any machine without a GPU. Key invariants covered:

- `keep_alive=0` is present in every council generate call.
- Judge call does **not** use `keep_alive=0`.
- Council rejects duplicate models.
- Production rejects test-only and unknown models.
- Session JSON matches the documented schema.

---

## Project layout

```
council/            Core logic — models allowlist, async orchestrator, rich output
setup/              Interactive installer: detect.py, plans.py, setup.py
api/                FastAPI + SSE web backend (server.py)
frontend/           Single-file dark UI (index.html)
tests/              Pytest suite with mocked Ollama transport
results/            Session JSON output (gitignored)
legacy/             Archived pre-migration Flask app
council.py          CLI entrypoint
Makefile            make setup / web / run / test / demo / results / clean
FUNCTIONAL_SPEC.md  The specification this implementation targets
ARCHITECTURE.md     High-level architectural overview
```

See [FUNCTIONAL_SPEC.md](FUNCTIONAL_SPEC.md) for the detailed functional contract and [ARCHITECTURE.md](ARCHITECTURE.md) for the shareable overview.

---

© Vishaka Datta J Hebbar 2026
