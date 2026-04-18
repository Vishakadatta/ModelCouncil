# Council vs Giant — Functional & Technical Specification

> **Version:** 1.1
> **Last updated:** 2026-03-25
> **Target hardware:** MacBook M2, 8GB RAM
> **Runtime:** Python 3.10+, Ollama, Flask

---

## 1. What This Project Does

Council vs Giant answers one question: **Can a group of tiny AI models, working together and judged by a bigger model, match that bigger model working alone?**

The system sends the same question to 3 small local LLMs (the "Council") and 1 larger local LLM (the "Giant"). Then they **judge each other** — the giant evaluates the council's answers, or the council evaluates the giant's answer. This cross-evaluation is the core value: you don't just see answers, you see reasoned verdicts on who got it right and wrong.

There are two interfaces:
- **Web UI** (`app.py`) — interactive, type any question, pick a mode, see color-coded result cards
- **CLI Benchmark** (`benchmark.py`) — automated, runs 20 hardcoded factual questions, scores everything, prints a comparison table

**Every query is automatically persisted** to a local SQLite database (`history.db`). A dedicated history page lets you browse, search, and replay any past query with full responses.

---

## 2. Models & Hardware Constraints

### Installed Models

| Role | Model | Parameters | Disk Size | Purpose |
|------|-------|-----------|-----------|---------|
| Council | `qwen2.5:0.5b` | 0.5B | 397 MB | Tiny — fast but error-prone |
| Council | `tinyllama` | 1.1B | 637 MB | Tiny — often rambles |
| Council | `gemma2:2b` | 2B | 1.6 GB | Small — usually the most accurate council member |
| Giant | `phi3:mini` | 3.8B | 2.2 GB | The "big" model — judge and solo answerer |

### Why These Sizes

8GB RAM is shared between macOS and Ollama. Realistic usable VRAM is ~5-6GB. The system must:
- Never load all 4 models simultaneously
- Load/unload aggressively between phases
- Run the 3 council models in parallel (they're small enough to coexist briefly)
- Run the giant model alone

All model names are configured in `council/config.py`. To swap models, edit that single file.

---

## 3. Architecture Overview

```
                         ┌─────────────────────────────────────────────┐
                         │                  User                       │
                         │         (Browser or Terminal)                │
                         └──────────┬──────────────┬──────────────────┘
                                    │              │
                            Web UI (app.py)   CLI (benchmark.py)
                                    │              │
                         ┌──────────▼──────────────▼──────────────────┐
                         │           council/ package                   │
                         │                                             │
                         │  config.py    ── endpoints, model names     │
                         │  ollama.py    ── HTTP calls, async wrapper  │
                         │  engine.py    ── 4 execution modes + judging│
                         │  judge.py     ── 0-10 scoring (bench only)  │
                         │  questions.py ── 20 benchmark Q&A pairs     │
                         │  storage.py   ── SQLite persistence layer   │
                         └────────┬─────────────────┬─────────────────┘
                                  │                 │
                         HTTP (localhost:11434)   SQLite (history.db)
                                  │                 │
                         ┌────────▼─────────┐  ┌───▼──────────────────┐
                         │  Ollama Server    │  │  history.db          │
                         │  Model inference  │  │  All queries + resp  │
                         │  VRAM management  │  │  Persistent, local   │
                         └──────────────────┘  └──────────────────────┘
```

---

## 4. Directory Structure

```
ModelCouncil/
├── council/                    # Core library (reusable, no Flask dependency)
│   ├── __init__.py             # Package marker
│   ├── config.py               # All configuration in one place
│   ├── ollama.py               # Low-level Ollama HTTP + async helpers
│   ├── engine.py               # Execution modes: all, council, giant, single
│   ├── judge.py                # Numeric scoring (used by benchmark only)
│   ├── questions.py            # 20 benchmark questions with ground truth
│   └── storage.py              # SQLite persistence — auto-logs every query
│
├── templates/
│   ├── index.html              # Main web UI — ask questions, see results
│   └── history.html            # History browser — table, stats, detail modal
│
├── app.py                      # Flask web server (entry point for web UI)
├── benchmark.py                # CLI benchmark runner (entry point for CLI)
├── council_vs_giant.py         # Original monolithic script (kept for reference)
├── history.db                  # SQLite database (auto-created on first run)
├── results.json                # Output from benchmark runs
└── SPEC.md                     # This file
```

### Module Dependency Graph

```
app.py ──────────► council/config.py
  │                council/ollama.py
  │                council/storage.py ──► history.db (SQLite)
  └──────────────► council/engine.py ──► council/ollama.py
                                         council/config.py

benchmark.py ────► council/config.py
  │                council/ollama.py
  │                council/engine.py
  └──────────────► council/judge.py ───► council/ollama.py
                   council/questions.py
```

---

## 5. Module Reference

### 5.1 `council/config.py` — Configuration

All tunable values live here. Nothing else in the codebase hardcodes model names or URLs.

```python
OLLAMA_BASE       = "http://localhost:11434"
OLLAMA_GENERATE   = "http://localhost:11434/api/generate"
OLLAMA_TAGS       = "http://localhost:11434/api/tags"
COUNCIL_MODELS    = ["qwen2.5:0.5b", "tinyllama", "gemma2:2b"]
GIANT_MODEL       = "phi3:mini"
GENERATE_TIMEOUT  = 300  # seconds per generation call
```

**To add or swap a model:** edit `COUNCIL_MODELS` or `GIANT_MODEL` here. Everything downstream reads from these constants.

---

### 5.2 `council/ollama.py` — Ollama HTTP Layer

Every interaction with Ollama goes through this file. No other module makes HTTP calls.

| Function | Signature | Returns | Description |
|----------|-----------|---------|-------------|
| `is_ollama_running()` | `() → bool` | `True`/`False` | Pings `GET /api/tags` with 5s timeout |
| `list_models()` | `() → list[str]` | `["phi3:mini", ...]` | Parses model names from `/api/tags` response |
| `generate(model, prompt)` | `(str, str) → str` | Response text | `POST /api/generate` with `stream: False`, 300s timeout |
| `unload(model)` | `(str) → None` | Nothing | `POST /api/generate` with `keep_alive: 0`, empty prompt. Silently catches all exceptions. |
| `async_generate(model, prompt)` | `async (str, str) → tuple[str, str]` | `(model_name, response)` | Runs `generate()` in a thread via `loop.run_in_executor` |

**Ollama API payload for generation:**
```json
{
  "model": "phi3:mini",
  "prompt": "What is 2+2?",
  "stream": false
}
```

**Ollama API payload for unloading:**
```json
{
  "model": "phi3:mini",
  "prompt": "",
  "stream": false,
  "keep_alive": 0
}
```

---

### 5.3 `council/engine.py` — Execution Modes

This is the brain. It orchestrates which models run, in what order, and who judges whom.

#### Internal Helpers (not exported)

```python
async _query_council_parallel(question: str) → dict[str, str]
```
Fires all 3 council models concurrently using `asyncio.gather`. Returns `{"qwen2.5:0.5b": "answer", "tinyllama": "answer", "gemma2:2b": "answer"}`.

```python
async _council_judge_parallel(giant_answer: str, question: str) → dict[str, str]
```
Each council model judges the giant's answer concurrently. Returns `{"model": "judgment", ...}`.

```python
giant_judge_council(question: str, council_answers: dict) → str
```
Giant reads all 3 council answers, states CORRECT/WRONG for each, gives its own best answer.

#### Exported Mode Functions

**`query_single(model, question) → str`**

Just calls `generate()` directly. No judging.

---

**`run_council_mode(question) → dict`**

Council answers, Giant judges.

```
Step 1: Query 3 council models in parallel ─── asyncio
Step 2: Unload all 3 council models ────────── free VRAM
Step 3: Giant judges council answers ───────── single call
```

Returns:
```python
{
    "individual": {"qwen2.5:0.5b": "...", "tinyllama": "...", "gemma2:2b": "..."},
    "giant_judgment": "For qwen2.5:0.5b: CORRECT because... For tinyllama: WRONG because..."
}
```

---

**`run_giant_mode(question) → dict`**

Giant answers, Council judges.

```
Step 1: Giant answers the question ─────────── single call
Step 2: Unload giant ───────────────────────── free VRAM
Step 3: 3 council models judge in parallel ─── asyncio
Step 4: Unload council models ──────────────── free VRAM
```

Returns:
```python
{
    "giant_answer": "The answer is...",
    "council_judgments": {
        "qwen2.5:0.5b": "This is correct because...",
        "tinyllama": "The answer appears wrong...",
        "gemma2:2b": "Correct. The answer is..."
    }
}
```

---

**`run_all_mode(question) → dict`**

Everyone answers, Giant judges the council.

```
Step 1: Query 3 council models in parallel ─── asyncio
Step 2: Unload all 3 council models ────────── free VRAM
Step 3: Giant answers the question directly ── single call
Step 4: Giant judges all council answers ───── single call (giant still loaded)
```

Returns:
```python
{
    "council_individual": {"qwen2.5:0.5b": "...", "tinyllama": "...", "gemma2:2b": "..."},
    "giant_answer": "The answer is...",
    "giant_judgment": "For each model: CORRECT/WRONG... My best answer is..."
}
```

---

### 5.4 `council/judge.py` — Numeric Scoring

Used **only by the CLI benchmark**, not the web UI. The web UI uses natural-language judging instead.

```python
score_answer(ground_truth: str, answer: str) → int   # 0-10
```

Prompt sent to giant model:
```
Given the correct answer is {ground_truth}, score the following response
from 0 to 10 for factual accuracy only. Respond with just a number.
Response: {answer}
```

Score parsing logic (`_parse_score`):
1. Split response by whitespace
2. Strip punctuation: `".,;:!?/()[]**"`
3. Try `int(token)` — if 0-10, return it
4. Try `int(float(token))` — if 0-10, return it
5. If nothing works, return **5** as fallback

**Known issue:** The giant model sometimes writes verbose explanations instead of just a number, causing fallback to 5. This affects benchmark accuracy.

---

### 5.5 `council/questions.py` — Benchmark Questions

20 questions across 6 domains:

| Domain | Count | Example |
|--------|-------|---------|
| History | 3 | "What year did World War 2 end?" → "1945" |
| Science | 5 | "What element has the atomic number 1?" → "Hydrogen" |
| Math | 3 | "What is the square root of 144?" → "12" |
| Geography | 4 | "What is the capital of Australia?" → "Canberra" |
| Biology | 2 | "How many chromosomes do humans have?" → "46" |
| General | 3 | "Who painted the Mona Lisa?" → "Leonardo da Vinci" |

Stored as:
```python
BENCHMARK_QUESTIONS = [
    {"question": "...", "ground_truth": "..."},
    ...
]
```

---

### 5.6 `council/storage.py` — Persistent Query History

Every query made through the web UI is automatically saved to a SQLite database (`history.db` in the project root). The database is created automatically on first import.

#### Why SQLite

| Considered | Rejected because |
|------------|-----------------|
| Append-only JSON Lines | Not queryable, no structure, hard to paginate |
| Rolling text files | File management overhead, can't search across files |
| Single JSON array | Must rewrite entire file on each append, corruption-prone on crash |
| **SQLite (chosen)** | Single file, ships with Python, handles millions of rows, crash-safe (WAL mode), queryable |

#### Database Schema

```sql
CREATE TABLE history (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp         TEXT NOT NULL,        -- ISO 8601 datetime
    question          TEXT NOT NULL,        -- The user's question
    mode              TEXT NOT NULL,        -- "all", "council", "giant", "single"
    council_answers   TEXT,                 -- JSON dict of {model: answer} or NULL
    giant_answer      TEXT,                 -- Giant's direct response or NULL
    judgment          TEXT,                 -- Judgment text (giant or council JSON) or NULL
    elapsed_seconds   REAL,                -- Seconds taken for the full query
    raw_response      TEXT NOT NULL         -- Complete API response as JSON (everything)
);
```

The `raw_response` column stores the exact JSON returned by `/api/ask`, so no data is ever lost even if the structured columns don't capture every field.

#### Configuration

- **Database path:** `history.db` in the project root (set in `storage.py` as `DB_PATH`)
- **Journal mode:** WAL (Write-Ahead Logging) — safe for concurrent reads while Flask is writing
- **Auto-init:** Table is created on module import (`init_db()` runs at import time)

#### Functions

| Function | Signature | Returns | Description |
|----------|-----------|---------|-------------|
| `init_db()` | `() → None` | Nothing | Creates the `history` table if it doesn't exist. Called automatically on import. |
| `log_query(question, mode, response)` | `(str, str, dict) → int` | Row ID | Extracts structured fields from response dict, stores full JSON in `raw_response`. Returns the auto-increment ID. |
| `get_history(limit, offset)` | `(int, int) → list[dict]` | List of row dicts | Newest-first pagination. Default: 50 rows, offset 0. |
| `get_entry(entry_id)` | `(int) → dict or None` | Single row dict | Fetch one entry by primary key. Returns `None` if not found. |
| `get_stats()` | `() → dict` | Stats dict | Returns `{"total_queries": N, "by_mode": {"all": X, "council": Y, ...}}` |

#### How `log_query` maps response fields

| Mode | `council_answers` | `giant_answer` | `judgment` |
|------|-------------------|----------------|------------|
| `all` | JSON of `council_individual` | Giant's direct answer | Giant's judgment text |
| `council` | JSON of `individual` | NULL | Giant's judgment text |
| `giant` | NULL | Giant's direct answer | JSON of `council_judgments` |
| `single` | NULL | The model's answer | NULL |

#### Querying from terminal

Since it's a standard SQLite file, you can query it directly:

```bash
# See all questions asked
sqlite3 history.db "SELECT id, timestamp, question, mode FROM history ORDER BY id DESC"

# Find all council-mode queries
sqlite3 history.db "SELECT question, judgment FROM history WHERE mode='council'"

# Count queries per mode
sqlite3 history.db "SELECT mode, COUNT(*) FROM history GROUP BY mode"

# Export full history as JSON
sqlite3 -json history.db "SELECT * FROM history" > export.json
```

---

## 6. Prompt Templates

These are the exact prompts sent to models. They matter — changing wording changes behavior.

### Giant Judging Council
```
You are a senior judge reviewing answers from three smaller AI models.

Question: {question}

--- {model_name_1} ---
{answer_1}

--- {model_name_2} ---
{answer_2}

--- {model_name_3} ---
{answer_3}

For each model's response above:
1. State whether it is CORRECT or WRONG (and why, briefly)
2. Then give your own best, most accurate answer to the question.
```

### Council Judging Giant
```
You are reviewing another AI's answer for accuracy.

Question: {question}

Answer to review:
{giant_answer}

Judge this answer: Is it correct or wrong? Explain briefly,
then give your own best answer to the question.
```

### Numeric Scoring (benchmark only)
```
Given the correct answer is {ground_truth}, score the following response
from 0 to 10 for factual accuracy only. Respond with just a number.
Response: {answer}
```

---

## 7. Web UI — `app.py` + `templates/`

### Server

Flask app on **port 5050** (5000 conflicts with macOS AirPlay Receiver).

### Pages

| Route | Template | Purpose |
|-------|----------|---------|
| `GET /` | `index.html` | Main UI — ask questions, see results |
| `GET /history` | `history.html` | Browse all past queries with full responses |

### API Endpoints

#### `POST /api/ask`

**Request:**
```json
{
  "question": "What year did World War I end?",
  "mode": "all"
}
```

Mode values: `"all"`, `"council"`, `"giant"`, `"single:<model_name>"`

**Response by mode:**

Every response includes `elapsed` (seconds), `question`, and `history_id` (the SQLite row ID confirming it was saved).

| Mode | Additional fields |
|------|-------------------|
| `single` | `model, answer` |
| `council` | `individual, giant_judgment, council_models, giant_model` |
| `giant` | `giant_model, giant_answer, council_judgments, council_models` |
| `all` | `council_individual, giant_answer, giant_judgment, council_models, giant_model` |

**Example "all" mode response:**
```json
{
  "mode": "all",
  "question": "What is the capital of Australia?",
  "elapsed": 28.5,
  "history_id": 14,
  "council_individual": {
    "qwen2.5:0.5b": "The capital is Sydney...",
    "tinyllama": "Australia's capital city is Canberra...",
    "gemma2:2b": "The capital of Australia is **Canberra**..."
  },
  "giant_answer": "The capital of Australia is Canberra...",
  "giant_judgment": "qwen2.5:0.5b: WRONG — Sydney is not the capital...\ntinyllama: CORRECT...\ngemma2:2b: CORRECT...\nMy answer: Canberra.",
  "council_models": ["qwen2.5:0.5b", "tinyllama", "gemma2:2b"],
  "giant_model": "phi3:mini"
}
```

#### `GET /api/history`

Paginated query history, newest first.

**Query parameters:**
- `limit` (int, default 50) — max rows to return
- `offset` (int, default 0) — skip this many rows

**Response:**
```json
{
  "rows": [
    {
      "id": 14,
      "timestamp": "2026-03-25T14:32:01.123456",
      "question": "What is the capital of Australia?",
      "mode": "all",
      "council_answers": "{\"qwen2.5:0.5b\": \"...\", ...}",
      "giant_answer": "The capital is Canberra...",
      "judgment": "qwen2.5:0.5b: WRONG...",
      "elapsed_seconds": 28.5,
      "raw_response": "{...full JSON...}"
    }
  ],
  "limit": 50,
  "offset": 0
}
```

#### `GET /api/history/<id>`

Single entry with the `raw_response` parsed back into a dict.

**Response:**
```json
{
  "id": 14,
  "timestamp": "2026-03-25T14:32:01.123456",
  "question": "What is the capital of Australia?",
  "mode": "all",
  "council_answers": "{...}",
  "giant_answer": "...",
  "judgment": "...",
  "elapsed_seconds": 28.5,
  "raw_response": "{...}",
  "parsed_response": { ... full response dict ... }
}
```

The `parsed_response` field is the full original API response (same shape as `/api/ask` output), reconstructed from the stored JSON. This is what the history detail modal renders.

#### `GET /api/stats`

Aggregate statistics.

**Response:**
```json
{
  "total_queries": 47,
  "by_mode": {
    "all": 30,
    "council": 10,
    "giant": 5,
    "single": 2
  }
}
```

#### `GET /api/models`
Returns `{"models": ["phi3:mini", "gemma2:2b", "tinyllama:latest", "qwen2.5:0.5b"]}`.

---

### Frontend Architecture — `index.html` (Main Page)

Single-page app. No framework — vanilla JS with `fetch()`.

**Global state:**
- `currentMode` — string, tracks selected mode button
- `history` — in-memory array of `{q, elapsed}`, max 20 entries (session only, for quick re-asks; permanent history is in SQLite)

**Key functions:**

| Function | Purpose |
|----------|---------|
| `setMode(btn)` | Updates mode, toggles model dropdown visibility, updates description text |
| `askQuestion()` | Validates input, shows loading spinner with animated steps, POSTs to `/api/ask`, renders results |
| `renderResults(data)` | Builds HTML cards based on response mode. Shows `Saved #N` green tag from `history_id`. |
| `card(title, body, type, isJudge)` | Creates a single result card. `isJudge=true` adds gold border. |
| `sectionLabel(text, type)` | Colored uppercase section divider |
| `addToHistory(q, elapsed)` | Prepends to in-memory history array, renders clickable list |
| `reask(q)` | Re-runs a history question |
| `esc(s)` | HTML-escapes text for safe rendering |

**Card types and colors:**

| Tag | Color | Used for |
|-----|-------|----------|
| `council` | Purple (#6c5ce7) | Individual tiny model answers |
| `giant` | Pink (#fd79a8) | Giant model's direct answer |
| `judge` | Gold (#fdcb6e) | Any judgment card (giant or council judging) |
| `single` | Teal (#00cec9) | Single-model mode answer |

**Header elements:**
- Title + subtitle
- Status pills (Ollama connection + installed models)
- **"View Full History"** button linking to `/history`

**Meta bar** (shown above results after each query):
- Question text
- Elapsed time
- Mode used
- **"Saved #14"** green indicator (confirms SQLite persistence, shows row ID)

**Loading animation:**
Each mode has predefined steps shown during loading. Steps advance every ~6 seconds with a CSS animation. Steps transition from `pending` (gray) → `active` (purple) → `done` (green with `>>` prefix).

---

### Frontend Architecture — `history.html` (History Page)

Dedicated page at `/history` for browsing all past queries.

**Layout:**
1. **Stats bar** — cards showing total queries and count per mode (fetched from `GET /api/stats`)
2. **History table** — columns: `#`, `Question`, `Mode`, `Time`, `Date`
3. **Load More button** — pagination, 50 rows at a time (fetched from `GET /api/history`)
4. **Detail modal** — click any row to open a full-screen overlay with all response cards

**Key functions:**

| Function | Purpose |
|----------|---------|
| `loadStats()` | Fetches `/api/stats`, renders stat cards |
| `loadHistory()` | Fetches `/api/history` with pagination, appends table rows |
| `loadMore()` | Triggers next page of history |
| `showDetail(id)` | Fetches `/api/history/<id>`, renders full response in modal |
| `closeModal(e)` | Closes modal on overlay click or Escape key |
| `detailCard(title, body, isJudge)` | Renders a response card inside the modal |
| `detailLabel(text, type)` | Section label inside the modal |

**Detail modal rendering:**
The modal fetches the `parsed_response` from the API and renders the exact same card layout as the main page — council answers, giant answer, judgment cards — using the same color coding. This means clicking any historical entry shows exactly what you saw when you first asked the question.

**Mode tags in table:**
Each mode gets a colored pill matching the main page's color scheme (purple gradient for "all", purple for "council", pink for "giant", teal for "single").

---

## 8. CLI Benchmark — `benchmark.py`

### How to Run
```bash
python3 benchmark.py
```

### Execution Flow Per Question

```
[1/20] What year did World War 2 end?
  Ground truth: 1945
  Council mode...
    → 3 models answer in parallel (asyncio)
    → Unload council
    → Giant judges council answers (run_council_mode)
  Giant mode...
    → Giant answers directly (generate)
  Scoring...
    → Giant scores council judgment 0-10
    → Giant scores giant answer 0-10
  Time: 30.2s
```

### Imports

```python
from council.engine import run_council_mode, query_single
from council.ollama import generate
from council.judge import score_answer
from council.questions import BENCHMARK_QUESTIONS
```

### Output

**Terminal table:**
```
------------------------------------------------------------
#    Question                                     Council    Giant    Winner
------------------------------------------------------------
1    What year did World War 2 end?                    8        5    Council
2    What is the chemical symbol for gold?             5       10      Giant
...
```

Winner logic: score difference > 1 = winner, ≤ 1 = tie.

**Terminal summary:**
```
===== SUMMARY =====
Council average score : 7.8 / 10
Giant average score   : 8.2 / 10
Council wins          : 4
Giant wins            : 5
Ties (within 1 point) : 11
Overall winner: GIANT
```

**JSON output** saved to `results.json` with full answers and scores per question.

---

## 9. VRAM Management

This is critical on 8GB RAM. The system **never** tries to load all models at once.

### Timeline: All Mode

```
Time ──────────────────────────────────────────────────►

VRAM usage:

 ┌─ qwen:0.5b ─┐
 ┌─ tinyllama ──┐
 ┌─ gemma2:2b ──┐ unload ┌── phi3:mini ──────────────────┐
 │  PARALLEL    │  all 3  │  Giant answers  │ Giant judges │
 │  ~2.6 GB     │         │     ~2.2 GB     │  (still loaded)
 └──────────────┘         └──────────────────────────────-┘
```

### Timeline: Giant Mode

```
 ┌── phi3:mini ──┐ unload ┌─ qwen:0.5b ──┐
 │  Giant answers │        ┌─ tinyllama ──┐  unload
 │  ~2.2 GB      │        ┌─ gemma2:2b ──┐  all 3
 └───────────────┘        │  PARALLEL    │
                          │  judging     │
                          │  ~2.6 GB     │
                          └──────────────┘
```

### How Unloading Works

Ollama API with `keep_alive: 0` tells Ollama to immediately evict the model from memory after the request completes. The `unload()` function sends an empty prompt with this flag. Exceptions are silently caught — if unloading fails, Ollama's own eviction policy handles it (LRU).

---

## 10. Data Persistence

### What Gets Saved

Every query through the web UI is automatically logged via `log_query()` in the `/api/ask` endpoint. The CLI benchmark does **not** log to SQLite (it saves to `results.json` instead).

### Where It Lives

`history.db` — a single SQLite file in the project root. Auto-created on first run.

### How Much It Can Hold

SQLite comfortably handles millions of rows. At ~2KB per query (average response sizes), you'd need ~500,000 queries to hit 1GB. No rotation or cleanup needed.

### Backup

```bash
# Copy the file — it's a single file, that's the whole backup
cp history.db history_backup.db

# Or export to JSON
sqlite3 -json history.db "SELECT * FROM history" > history_export.json
```

### Data Flow

```
User asks question
       │
       ▼
  app.py /api/ask
       │
       ├──► engine.py (run models, get responses)
       │
       ├──► Build result dict with all answers + judgments
       │
       ├──► storage.log_query(question, mode, result)  ◄── AUTO-SAVE
       │         │
       │         ▼
       │    history.db INSERT
       │         │
       │         ▼
       │    Returns row_id
       │
       ▼
  result["history_id"] = row_id
       │
       ▼
  Return JSON to browser (includes history_id)
       │
       ▼
  Browser shows "Saved #14" green tag
```

---

## 11. Error Handling

| Scenario | Behavior |
|----------|----------|
| Ollama not running | Web: shows "Offline" pill, API returns 503. CLI: prints error, exits. |
| Model not installed | Ollama returns HTTP error, `generate()` raises `requests.HTTPError` |
| Model timeout (>300s) | `requests.post` raises `Timeout`, propagates to caller |
| Unload fails | Silently ignored (best-effort) |
| Score can't be parsed | Returns 5 as fallback, prints warning to terminal |
| Empty question | API returns 400 with `{"error": "No question provided"}` |
| Unknown mode | API returns 400 with `{"error": "Unknown mode: ..."}` |
| History entry not found | `/api/history/<id>` returns 404 with `{"error": "Not found"}` |
| SQLite write fails | Exception propagates (should not happen under normal conditions with WAL mode) |

---

## 12. How to Run

### Prerequisites
```bash
# Install Ollama
brew install ollama

# Start Ollama service
brew services start ollama

# Pull models (one-time)
ollama pull qwen2.5:0.5b
ollama pull tinyllama
ollama pull gemma2:2b
ollama pull phi3:mini

# Install Python dependencies
pip3 install flask requests
```

### Web UI
```bash
cd ModelCouncil
python3 app.py
# Open http://localhost:5050
# History at http://localhost:5050/history
```

### CLI Benchmark
```bash
cd ModelCouncil
python3 benchmark.py
# Runs ~10 minutes for 20 questions
# Results saved to results.json
```

---

## 13. Known Issues & Limitations

1. **Score parsing fragility** — The giant model often responds with explanations instead of a single number. The parser falls back to 5, skewing benchmark scores. Consider adding `"You MUST respond with only a single integer"` to the prompt.

2. **No streaming** — All API calls use `stream: False`, so the user sees nothing until the full response is ready. Could add SSE/streaming for better UX.

3. **Council parallel loading** — When 3 council models run in parallel, all 3 load into VRAM simultaneously. On 8GB this works because they're tiny, but would fail with larger council models.

4. **No request queuing** — If a user submits a second question while the first is still running, both will execute concurrently and compete for VRAM. The web UI disables the button, but nothing prevents direct API abuse.

5. **Judge bias** — The giant judges the council, so it naturally rates itself higher. In giant-only mode, tiny models judge the giant, which may be unreliable. This is a feature (seeing the bias), not a bug.

6. **CLI benchmark does not log to SQLite** — Only the web UI auto-saves to `history.db`. The CLI saves to `results.json` instead. These are separate data stores.

---

## 14. How to Contribute

### Adding a New Model to the Council
1. Pull it: `ollama pull <model_name>`
2. Edit `council/config.py`: add to `COUNCIL_MODELS` list
3. Test VRAM: ensure all council models + OS fit in 8GB simultaneously

### Adding a New Execution Mode
1. Add a new function in `council/engine.py` following the pattern of `run_*_mode()`
2. Add the route handler in `app.py` under `/api/ask`
3. Add rendering logic in `templates/index.html` inside `renderResults()`
4. Add the mode button and description in the HTML
5. Update `storage.log_query()` to extract the right fields for the new mode

### Adding Benchmark Questions
1. Add to `council/questions.py` → `BENCHMARK_QUESTIONS` list
2. Format: `{"question": "...", "ground_truth": "..."}`
3. Ground truth should be a short, unambiguous string

### Changing the Giant Model
1. Pull it: `ollama pull <model_name>`
2. Edit `council/config.py`: change `GIANT_MODEL`
3. Ensure it fits in VRAM alone (~3GB max for 8GB machines)

### Adding a New Storage Column
1. Add column to `CREATE TABLE` in `council/storage.py`
2. Update `log_query()` to populate it
3. Delete `history.db` and restart (or run `ALTER TABLE` manually)
4. Update `history.html` if the column should appear in the UI

### Running Tests Manually
```bash
# Quick smoke test — single model, fast
curl -X POST http://localhost:5050/api/ask \
  -H 'Content-Type: application/json' \
  -d '{"question":"What is 2+2?","mode":"single:qwen2.5:0.5b"}'

# Council mode
curl -X POST http://localhost:5050/api/ask \
  -H 'Content-Type: application/json' \
  -d '{"question":"What is the capital of France?","mode":"council"}'

# Check history was saved
curl -s http://localhost:5050/api/history?limit=5 | python3 -m json.tool

# View specific entry
curl -s http://localhost:5050/api/history/1 | python3 -m json.tool

# Check stats
curl -s http://localhost:5050/api/stats | python3 -m json.tool

# Full benchmark
python3 benchmark.py

# Query SQLite directly
sqlite3 history.db "SELECT id, question, mode, elapsed_seconds FROM history ORDER BY id DESC LIMIT 10"
```
