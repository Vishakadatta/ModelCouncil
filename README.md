# Model Council

> **The only AI council you can try in your browser. No install. No GPU. No signup.**

🌐 **[→ Open Model Council Live](https://vishakadatta.github.io/ModelCouncil/)**

Type any question. Three small AI models answer in parallel. A larger judge model reviews their answers and delivers a verdict. All in your browser, in seconds.

Powered by [NVIDIA's free hosted GPU API](https://build.nvidia.com).

---

## Why this is different

Most "model council" projects on GitHub require you to:
- Install Ollama or LM Studio
- Pull 5–10 GB of model weights
- Have a GPU
- Run `make` / `npm` / `python setup.py`
- Then maybe see something work

**Model Council removes all of that.** Click the link, ask a question, see four AI models reason in real time. Live demo, no friction.

If you want to run it yourself, that's also a one-command install. But you don't have to — that's the point.

---

## What it does

```
Your question
      │
      ▼
┌─────────────────────────────────────────────┐
│  Council — 3 small models, run in parallel  │
│                                             │
│  Llama 3.1 8B   ·   Mistral 7B   ·   Gemma  │
└─────────────────────┬───────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────┐
│  Judge — NVIDIA Nemotron 70B                │
│                                             │
│  Phase 1: answers independently             │
│  Phase 2: reviews each council answer,      │
│           tags CORRECT / WRONG / PARTIAL,   │
│           issues final authoritative answer │
└─────────────────────────────────────────────┘
```

The judge thinks **before** seeing the council's answers, then critiques them. This two-phase design stops the judge from rubber-stamping the council when it's wrong — which is the failure mode in naive "best of 3" approaches.

---

## Run it your way

### 1. Browser only (free NVIDIA cloud)

**[https://vishakadatta.github.io/ModelCouncil/](https://vishakadatta.github.io/ModelCouncil/)**

Done. That's it.

### 2. Run the same thing locally on NVIDIA's free API

```bash
git clone https://github.com/Vishakadatta/ModelCouncil
cd ModelCouncil
pip install -r requirements.txt
make setup
# → wizard asks: NVIDIA NIM (cloud) or Ollama (local)?
# → choose NIM, paste your free key from build.nvidia.com
# → browser opens automatically
```

Get a free NVIDIA API key here: **[build.nvidia.com](https://build.nvidia.com)**.

### 3. Run it fully offline with your own GPU

```bash
make setup
# → choose Ollama
# → wizard detects your VRAM and picks a plan that fits
```

### 4. Try the smallest demo on a CPU laptop

```bash
make demo
make web
```

No GPU, no API key, no internet after the initial model pull.

---

## Architecture in one picture

```
GitHub Pages (frontend, static HTML)
     │
     │ HTTPS + Server-Sent Events
     ▼
Render (FastAPI, Python)
     │
     │ /v1/chat/completions
     ▼
NVIDIA NIM (free hosted GPU API)
     │
     ▼
Llama · Mistral · Gemma · Nemotron
```

Production deployment is automatic — every push to `main` redeploys both layers.

---

## Highlights

- **Two-phase deliberative judge** — independent answer first, then review
- **Live model discovery** — the setup wizard queries NVIDIA's catalogue and selects models automatically based on size and origin
- **Geographic diversity** — councils mix US and European labs (Meta, Mistral, Google) for varied reasoning styles
- **Auto-fallback on rate limits** — if a model is busy, the system silently retries with the next available one (13 council fallbacks, 5 judge fallbacks)
- **Streaming UI** — every token appears as it's generated, no waiting screens
- **Single HTML file frontend** — no build step, no npm, no framework
- **Two backends from one orchestrator** — same pipeline drives NVIDIA NIM (cloud) and Ollama (local)
- **Zero database** — sessions saved as JSON files, served from disk

---

## Make commands

| Command | What it does |
|---------|--------------|
| `make setup` | Interactive wizard — choose NIM or Ollama, configure everything |
| `make web` | Start the local server on `http://localhost:7860` |
| `make demo` | CPU-only demo with tiny models, no API key needed |
| `make test` | Run the test suite (no GPU, no network) |
| `make results` | Pretty-print the latest session JSON |

---

## Project layout

```
api/              FastAPI server + SSE session runner
council/          Core logic: orchestrator, NIM client, model policy
setup/            Setup wizard + live model discovery from NVIDIA
frontend/         Single-file web UI
tests/            14 tests, no GPU or network needed
results/          Session JSON output
.github/          CI/CD — auto-deploys frontend to GitHub Pages
render.yaml       Render service definition
Makefile          All commands

docs/             Technical documentation
  SPEC.md           Full technical specification
  ARCHITECTURE.md   Architecture overview
  FUNCTIONAL_SPEC.md Functional contract
cli.py            Pure-CLI entry point
```

---

## Live infrastructure

| Layer | Where | URL |
|-------|-------|-----|
| Frontend | GitHub Pages | https://vishakadatta.github.io/ModelCouncil/ |
| Backend API | Render | https://llm-modelcouncil.onrender.com |
| Model inference | NVIDIA NIM | https://build.nvidia.com |

---

## Tech stack

- **Backend:** Python 3.10+, FastAPI, asyncio, httpx, Server-Sent Events
- **Frontend:** Vanilla JS, EventSource API, single HTML file
- **Models (cloud):** NVIDIA NIM — Meta Llama, Mistral, Google Gemma, NVIDIA Nemotron
- **Models (local):** Ollama — any compatible model
- **Deployment:** Render (backend), GitHub Pages (frontend), GitHub Actions (CI/CD)
- **Testing:** pytest with `httpx.MockTransport`

---

## Why I built this

Single LLMs are confident even when wrong. Multiple smaller LLMs working together — and being judged by a larger one — surface disagreements you'd otherwise miss. This is a working demonstration that a 3-model council plus a deliberative judge can produce more honest reasoning than any single model alone, while staying small enough to run on commodity hardware or NVIDIA's free tier.

---

## Open source

MIT licensed. Fork it, deploy your own, share it.

If you build something interesting on top, open an issue — I'd love to see it.

---

© Vishaka Datta J Hebbar
