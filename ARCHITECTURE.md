# Architecture Overview

A shareable high-level view of **Model Council** — suitable for showing to another LLM or a reviewer without requiring them to read the full spec.

## The problem

Running large local LLMs is VRAM-bound. A 24 GB GPU cannot hold three 8B-class council models **and** a 27B judge model simultaneously. Naive parallel orchestration OOMs. Naive sequential orchestration is slow because each model reload takes seconds.

## The solution

Use Ollama's `keep_alive` parameter as a VRAM scheduler.

1. Fire all council models **in parallel** via `asyncio.gather`. Each request sets `keep_alive=0`.
2. The moment each council response completes, Ollama evicts that model from VRAM.
3. Once all three council models have finished, VRAM is empty.
4. Load the judge alone. It fits because nothing else is resident.
5. Peak VRAM = `max(sum(council), judge)`, not `sum(council) + judge`.

This single flag is what makes the whole design feasible on consumer hardware.

## The deliberative judge

The judge does not just synthesize. It deliberates in two phases, both streamed into the same output panel:

1. **Independent answer.** Judge sees only the user's question, answers from its own knowledge. Stays hot via `keep_alive="5m"`.
2. **Review + verdict.** Judge now sees the council's three answers plus its own Phase 1 output. For each council answer it tags CORRECT / PARTIALLY CORRECT / WRONG, absorbs genuine improvements, and issues a final authoritative verdict.

This defends against the common failure mode where a judge rubber-stamps the council even when the council is wrong.

## Component map

```
┌─────────────────────────────────────────────────────────────┐
│  Entry points                                               │
│    council.py          — CLI                                │
│    api/server.py       — FastAPI + SSE on :7860             │
│    frontend/index.html — single-file web UI                 │
│    setup/setup.py      — interactive installer              │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  council/orchestrator.py                                    │
│    Orchestrator — validates plan, runs council concurrently,│
│    streams judge. Agnostic to CLI vs. web.                  │
│    _stream_generate — async generator over Ollama NDJSON.   │
│    COUNCIL_KEEP_ALIVE = 0 — enforced constant.              │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  council/models.py                                          │
│    APPROVED_MODELS — production allowlist (5 vendors).      │
│    TEST_ONLY_MODELS — demo-only bucket.                     │
│    Policy checks: production allow, diverse origins.        │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
                     HTTP → Ollama daemon
```

## Data flow (web path)

```
Browser                FastAPI                    Ollama
   │   POST /ask          │                          │
   │─────────────────────>│                          │
   │  {session_id}        │                          │
   │<─────────────────────│                          │
   │                      │                          │
   │  GET /stream/{sid}   │                          │
   │─ SSE ───────────────>│                          │
   │                      │  /api/generate ×3 ║      │
   │                      │  (keep_alive=0)   ║──────│ parallel
   │ council_chunk ×N     │                   ║      │
   │<─────────────────────│<══════════════════╣      │
   │ council_done ×3      │                          │
   │                      │  /api/generate (judge,   │
   │                      │  phase 1, keep_alive=5m) │
   │ judge_chunk (phase1) │──────────────────────────│
   │<─────────────────────│<─────────────────────────│
   │                      │  /api/generate (judge,   │
   │                      │  phase 2, keep_alive=5m) │
   │ judge_chunk (phase2) │──────────────────────────│
   │<─────────────────────│<─────────────────────────│
   │ judge_done           │                          │
   │                      │                          │
   │                      │  writes results/*.json   │
```

## Invariants and how they are enforced

| Invariant                                           | Enforcement                                          |
|-----------------------------------------------------|------------------------------------------------------|
| `keep_alive=0` on every council call                | `test_keep_alive_zero_in_every_council_call`         |
| Judge does not use `keep_alive=0`                   | `test_judge_call_does_not_use_keep_alive_zero`       |
| Council members come from distinct vendors          | `assert_council_diverse` + constructor check         |
| Production plan uses only approved models           | `assert_production_allowed` + `ModelPolicyError`     |
| Tiny test models never leak into production         | Separate `TEST_ONLY_MODELS` list, production flag    |
| Judge is strictly larger than every council member  | Policy in `council/models.py`, plan-builder check    |
| Peak VRAM ≤ budget                                  | `setup/plans.py` refuses to render overweight plans  |

## Design choices worth calling out

- **Orchestrator is transport-agnostic.** The CLI and the web layer both call `Orchestrator._stream_generate`. The web layer wraps chunks in SSE events; the CLI prints them. No duplicate logic.
- **Frontend is one HTML file.** No build step, no npm, no framework. Vanilla JS + EventSource. Makes the demo trivial to share.
- **Hardcoded vendor allowlist.** Supply-chain and data-provenance constraints. Adding a vendor requires editing code and is a deliberate policy act.
- **Session JSON is the source of truth.** Both the CLI and the web layer write the same schema to `results/`. The history view reads the same files.
- **No database.** History is just a directory of JSON files sorted by mtime.

## What this codebase is not

- Not a chat product — each session is one question, one answer.
- Not a general agent framework — no tools, no memory, no planning.
- Not a benchmarking harness — it produces a single best answer, not a leaderboard.

## Project layout

```
council/          orchestrator, models allowlist, rich CLI output
setup/            interactive installer: detect, plans, setup
api/              FastAPI + SSE backend
frontend/         single-file dark UI
tests/            pytest + httpx.MockTransport, no GPU needed
results/          session JSON (gitignored)
legacy/           archived Flask app, not maintained
council.py        CLI entrypoint
Makefile          setup / web / run / test / demo / results / clean
README.md         user-facing guide
FUNCTIONAL_SPEC.md detailed functional contract
ARCHITECTURE.md   this file
```

---

© Vishaka Datta J Hebbar 2026
