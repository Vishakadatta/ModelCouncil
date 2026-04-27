# Model Council

**Can a group of small AI models, working together, match a single large model?**

Model Council sends your question to 3 small models simultaneously, then a larger judge model answers it independently — and then reviews the council's answers to produce a final verdict.

---

## Try it now — no install needed

**[→ Open Model Council](https://vishakadatta.github.io/ModelCouncil/)**

Runs entirely in your browser. Backed by NVIDIA's hosted models (NIM). No GPU, no account, no setup.

---

## How it works

```
Your question
      │
      ▼
┌─────────────────────────────────────────────┐
│  Council — 3 small models, run in parallel  │
│                                             │
│  Meta Llama 3.1 8B  (USA)                  │
│  Mistral 7B          (France)               │
│  Google Gemma 3 4B   (USA)                  │
└─────────────────────┬───────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────┐
│  Judge — NVIDIA Nemotron 70B                │
│                                             │
│  Phase 1: answers independently             │
│           (no council answers visible yet)  │
│                                             │
│  Phase 2: reviews each council answer,      │
│           tags CORRECT / WRONG / PARTIAL,   │
│           issues final authoritative answer │
└─────────────────────────────────────────────┘
```

The judge deliberates in **two phases** — it forms its own view first, then checks the council. This prevents the judge from being swayed by wrong council answers before thinking for itself.

All 3 council models stream in parallel. The judge panel appears once every council member finishes.

---

## Run it yourself

**Option A — Use NVIDIA NIM (no GPU needed, free tier)**

```bash
git clone https://github.com/Vishakadatta/ModelCouncil
cd ModelCouncil
pip install -r requirements.txt
make setup
# → wizard asks: Ollama or NIM?
# → choose NIM, paste your API key from build.nvidia.com
# → models are discovered live, plan is selected automatically
# → browser opens on http://localhost:7860
```

Get a free NVIDIA API key at: **[build.nvidia.com](https://build.nvidia.com)**

**Option B — Run fully local with Ollama (GPU recommended)**

```bash
make setup
# → choose Local (Ollama)
# → wizard detects your VRAM, picks a model plan that fits
# → pulls models, starts server, opens browser
```

**Option C — Demo mode (CPU-only, tiny models, no key needed)**

```bash
make demo
make web
```

Uses the smallest available models. Slow but works on any machine.

---

## Model selection

### NIM backend (cloud)

Models are discovered live from NVIDIA's catalogue at setup time — nothing is hardcoded. The wizard applies these rules automatically:

| Rule | What it does |
|------|-------------|
| Block Chinese-origin publishers | Alibaba/Qwen, DeepSeek, ByteDance, MiniMax, Moonshot/Kimi, Zhipu, Baidu, Step, BAAI, and others are never selected |
| Unknown origin → reject | Only publishers with verified western origin are trusted |
| Council: models < 15B | Small enough to run in parallel |
| Judge: models ≥ 30B | Large enough to reason well |
| Geographic diversity | Council must include ≥1 USA and ≥1 non-USA model |

If a model hits 503 (NVIDIA's free tier is busy) or 404 (model deprecated), the system automatically retries from a fallback pool of 13 council candidates and 5 judge candidates — silently, without failing the request.

### Ollama backend (local)

Uses a hardcoded allowlist of approved models (Meta, Mistral AI, Google, Microsoft, Cohere). The setup wizard detects your VRAM and selects a plan that fits.

---

## Make commands

```bash
make setup    # Interactive wizard — choose NIM or Ollama, configures everything
make web      # Start the server (requires .env from setup)
make demo     # CPU-only demo with tiny models, no key needed
make test     # Run the test suite (no GPU or network needed)
make results  # Pretty-print the latest session JSON
make clean    # Remove generated files
```

---

## Architecture

```
GitHub Pages (static HTML)
  frontend/index.html
  window.COUNCIL_API_URL → Render or localhost
          │
          │ HTTPS + Server-Sent Events
          ▼
FastAPI server (api/server.py)
  POST /ask      → creates session, spawns async pipeline
  GET /stream/   → SSE token stream to browser
  GET /health    → backend + model status
  GET /history   → last 10 sessions
          │
          ▼
council/orchestrator.py
  → routes to NIM or Ollama based on BACKEND env var
  → council: asyncio.gather (parallel)
  → judge: two sequential streaming calls
          │
          ▼
NVIDIA NIM  https://integrate.api.nvidia.com/v1/chat/completions
   OR
Ollama      http://localhost:11434/api/generate
```

Sessions are saved as JSON files in `results/`. The history panel in the UI reads these directly.

---

## Project layout

```
api/              FastAPI server + SSE session runner
council/          Core logic: orchestrator, NIM client, model policy
setup/            Interactive setup wizard + NIM model discovery
frontend/         Single-file web UI (no build step, no npm)
tests/            14 tests, no GPU or network needed
results/          Session JSON output (gitignored)
.github/          CI/CD — deploys frontend to GitHub Pages on push
render.yaml       Render service definition
Makefile          All commands
SPEC.md           Full technical specification
ARCHITECTURE.md   Architecture overview
FUNCTIONAL_SPEC.md Functional contract
```

---

## Deployment

The live site runs as two separate services:

| Layer | Platform | URL |
|-------|----------|-----|
| Frontend | GitHub Pages | https://vishakadatta.github.io/ModelCouncil/ |
| Backend API | Render (free tier) | https://llm-modelcouncil.onrender.com |

Every push to `main` automatically redeploys both. The GitHub Actions workflow injects the Render URL into the frontend before deploying to Pages.

---

## Open source

MIT licensed. Fork it, run it, modify it. If you run `make setup` and choose NIM, you're up in under 5 minutes with any machine — no GPU required.

---

© Vishaka Datta J Hebbar 2026
