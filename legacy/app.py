#!/usr/bin/env python3
"""
Council vs Giant - Interactive Web UI

Run with: python3 app.py
Then open: http://localhost:5050
"""

import json
import time
from flask import Flask, render_template, request, jsonify

from council.config import COUNCIL_MODELS, GIANT_MODEL
from council.ollama import is_ollama_running, list_models
from council.engine import query_single, run_council_mode, run_giant_mode, run_all_mode
from council.storage import log_query, get_history, get_entry, get_stats

app = Flask(__name__)


@app.route("/")
def index():
    ollama_ok = is_ollama_running()
    models = list_models() if ollama_ok else []
    return render_template(
        "index.html",
        ollama_ok=ollama_ok,
        models=models,
        council_models=COUNCIL_MODELS,
        giant_model=GIANT_MODEL,
    )


@app.route("/history")
def history_page():
    """Full history page."""
    return render_template("history.html")


@app.route("/api/ask", methods=["POST"])
def api_ask():
    """
    Accepts JSON: {question, mode}
    mode = "single:<model>" | "giant" | "council" | "all"
    """
    data = request.get_json()
    question = data.get("question", "").strip()
    mode = data.get("mode", "all")

    if not question:
        return jsonify({"error": "No question provided"}), 400

    start = time.time()

    if mode.startswith("single:"):
        model = mode.split(":", 1)[1]
        answer = query_single(model, question)
        result = {"mode": "single", "model": model, "answer": answer}

    elif mode == "giant":
        data = run_giant_mode(question)
        result = {
            "mode": "giant",
            "giant_model": GIANT_MODEL,
            "giant_answer": data["giant_answer"],
            "council_judgments": data["council_judgments"],
            "council_models": COUNCIL_MODELS,
        }

    elif mode == "council":
        data = run_council_mode(question)
        result = {
            "mode": "council",
            "individual": data["individual"],
            "giant_judgment": data["giant_judgment"],
            "council_models": COUNCIL_MODELS,
            "giant_model": GIANT_MODEL,
        }

    elif mode == "all":
        data = run_all_mode(question)
        result = {
            "mode": "all",
            "council_individual": data["council_individual"],
            "giant_answer": data["giant_answer"],
            "giant_judgment": data["giant_judgment"],
            "council_models": COUNCIL_MODELS,
            "giant_model": GIANT_MODEL,
        }

    else:
        return jsonify({"error": f"Unknown mode: {mode}"}), 400

    result["elapsed"] = round(time.time() - start, 1)
    result["question"] = question

    # Auto-save to SQLite
    row_id = log_query(question, result["mode"], result)
    result["history_id"] = row_id

    return jsonify(result)


# ---- History API ----

@app.route("/api/history")
def api_history():
    """Return paginated history. ?limit=50&offset=0"""
    limit = request.args.get("limit", 50, type=int)
    offset = request.args.get("offset", 0, type=int)
    rows = get_history(limit=limit, offset=offset)
    return jsonify({"rows": rows, "limit": limit, "offset": offset})


@app.route("/api/history/<int:entry_id>")
def api_history_entry(entry_id):
    """Return a single history entry with full raw response."""
    entry = get_entry(entry_id)
    if not entry:
        return jsonify({"error": "Not found"}), 404
    # Parse raw_response back to dict for the frontend
    entry["parsed_response"] = json.loads(entry["raw_response"])
    return jsonify(entry)


@app.route("/api/stats")
def api_stats():
    """Return aggregate stats."""
    return jsonify(get_stats())


@app.route("/api/models")
def api_models():
    if not is_ollama_running():
        return jsonify({"error": "Ollama not running"}), 503
    return jsonify({"models": list_models()})


if __name__ == "__main__":
    if not is_ollama_running():
        print("WARNING: Ollama is not running. Start it with: ollama serve")
    else:
        print(f"Ollama connected. Models: {list_models()}")

    print("\nStarting Council vs Giant Web UI...")
    print("Open http://localhost:5050 in your browser\n")
    app.run(debug=True, port=5050)
