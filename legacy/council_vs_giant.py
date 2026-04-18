#!/usr/bin/env python3
"""
Council vs Giant: Local LLM Benchmarking System

Tests whether a council of tiny models (qwen2.5:0.5b + tinyllama + gemma2:2b)
can match a single larger model (phi3:mini 3.8B) on factual questions.
Sized for MacBook M2 with 8GB RAM.
"""

import asyncio
import json
import time
import sys
import requests

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
OLLAMA_URL = "http://localhost:11434/api/generate"

COUNCIL_MODELS = ["qwen2.5:0.5b", "tinyllama", "gemma2:2b"]
GIANT_MODEL = "phi3:mini"

# ---------------------------------------------------------------------------
# Test questions with verified ground truth
# ---------------------------------------------------------------------------
QUESTIONS = [
    {"question": "What year did World War 2 end?", "ground_truth": "1945"},
    {"question": "What is the chemical symbol for gold?", "ground_truth": "Au"},
    {"question": "How many bones are in the adult human body?", "ground_truth": "206"},
    {"question": "What planet is closest to the sun?", "ground_truth": "Mercury"},
    {"question": "What is the speed of light in kilometers per second?", "ground_truth": "299792"},
    {"question": "Who wrote the play Romeo and Juliet?", "ground_truth": "William Shakespeare"},
    {"question": "What is the capital of Australia?", "ground_truth": "Canberra"},
    {"question": "What is the square root of 144?", "ground_truth": "12"},
    {"question": "What element has the atomic number 1?", "ground_truth": "Hydrogen"},
    {"question": "In what year did the Berlin Wall fall?", "ground_truth": "1989"},
    {"question": "What is the largest ocean on Earth?", "ground_truth": "Pacific Ocean"},
    {"question": "How many sides does a hexagon have?", "ground_truth": "6"},
    {"question": "What is the boiling point of water in Celsius?", "ground_truth": "100"},
    {"question": "Who painted the Mona Lisa?", "ground_truth": "Leonardo da Vinci"},
    {"question": "What is the smallest prime number?", "ground_truth": "2"},
    {"question": "What continent is Egypt in?", "ground_truth": "Africa"},
    {"question": "What gas do plants absorb from the atmosphere?", "ground_truth": "Carbon dioxide"},
    {"question": "How many chromosomes do humans have?", "ground_truth": "46"},
    {"question": "What is the currency of Japan?", "ground_truth": "Yen"},
    {"question": "What is the tallest mountain in the world?", "ground_truth": "Mount Everest"},
]

# ---------------------------------------------------------------------------
# Ollama helpers
# ---------------------------------------------------------------------------

def ollama_generate(model: str, prompt: str) -> str:
    """Send a prompt to an Ollama model and return the full response text."""
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
    }
    resp = requests.post(OLLAMA_URL, json=payload, timeout=300)
    resp.raise_for_status()
    return resp.json().get("response", "").strip()


def ollama_unload(model: str) -> None:
    """Unload a model from VRAM by setting keep_alive to 0."""
    payload = {
        "model": model,
        "prompt": "",
        "stream": False,
        "keep_alive": 0,
    }
    try:
        requests.post(OLLAMA_URL, json=payload, timeout=60)
    except Exception:
        pass  # best-effort unload


async def async_generate(model: str, prompt: str) -> tuple[str, str]:
    """Run ollama_generate in a thread so we can run multiple models concurrently."""
    loop = asyncio.get_event_loop()
    response = await loop.run_in_executor(None, ollama_generate, model, prompt)
    return model, response


# ---------------------------------------------------------------------------
# Core benchmark logic
# ---------------------------------------------------------------------------

async def run_council(question: str) -> tuple[dict[str, str], str]:
    """
    Council mode:
    1. Query all council models in parallel
    2. Unload small models
    3. Synthesize answers with the giant model
    Returns (individual_answers_dict, synthesized_answer).
    """
    # Step 1: query council in parallel
    tasks = [async_generate(m, question) for m in COUNCIL_MODELS]
    results = await asyncio.gather(*tasks)
    answers = {model: answer for model, answer in results}

    # Step 2: unload small models
    for model in COUNCIL_MODELS:
        ollama_unload(model)

    # Step 3: synthesize with giant model
    synthesis_prompt = (
        "You are a judge synthesizing multiple AI responses into one best answer.\n\n"
        f"Question: {question}\n\n"
    )
    for model, answer in answers.items():
        synthesis_prompt += f"Response from {model}:\n{answer}\n\n"
    synthesis_prompt += (
        "Based on these responses, provide the single best, most accurate answer "
        "to the question. Be concise."
    )

    synthesized = ollama_generate(GIANT_MODEL, synthesis_prompt)
    return answers, synthesized


def run_giant(question: str) -> str:
    """Giant mode: query the large model directly."""
    return ollama_generate(GIANT_MODEL, question)


def judge_answer(ground_truth: str, answer: str) -> int:
    """Use the giant model to score an answer 0-10 for factual accuracy."""
    judge_prompt = (
        f"Given the correct answer is {ground_truth}, score the following response "
        f"from 0 to 10 for factual accuracy only. Respond with just a number. "
        f"Response: {answer}"
    )
    raw = ollama_generate(GIANT_MODEL, judge_prompt)
    # Extract the first number from the response
    for token in raw.split():
        cleaned = token.strip(".,;:!?/()[]")
        try:
            score = int(cleaned)
            if 0 <= score <= 10:
                return score
        except ValueError:
            try:
                score = int(float(cleaned))
                if 0 <= score <= 10:
                    return score
            except ValueError:
                continue
    # Fallback: couldn't parse a score
    print(f"    [warn] Could not parse judge score from: {raw!r}, defaulting to 5")
    return 5


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def print_table(results: list[dict]) -> None:
    """Print a formatted results table."""
    q_width = 50
    header = f"{'#':<4} {'Question':<{q_width}} {'Council':>8} {'Giant':>8} {'Winner':>10}"
    sep = "-" * len(header)

    print("\n" + sep)
    print(header)
    print(sep)

    for i, r in enumerate(results, 1):
        q = r["question"]
        if len(q) > q_width - 3:
            q = q[: q_width - 3] + "..."
        cs = r["council_score"]
        gs = r["giant_score"]
        diff = cs - gs
        if diff > 1:
            winner = "Council"
        elif diff < -1:
            winner = "Giant"
        else:
            winner = "Tie"
        print(f"{i:<4} {q:<{q_width}} {cs:>8} {gs:>8} {winner:>10}")

    print(sep)


def print_summary(results: list[dict]) -> None:
    """Print aggregate statistics."""
    council_scores = [r["council_score"] for r in results]
    giant_scores = [r["giant_score"] for r in results]

    council_avg = sum(council_scores) / len(council_scores)
    giant_avg = sum(giant_scores) / len(giant_scores)

    council_wins = [r for r in results if r["council_score"] - r["giant_score"] > 1]
    giant_wins = [r for r in results if r["giant_score"] - r["council_score"] > 1]
    ties = [r for r in results if abs(r["council_score"] - r["giant_score"]) <= 1]

    print("\n===== SUMMARY =====")
    print(f"Council average score : {council_avg:.1f} / 10")
    print(f"Giant average score   : {giant_avg:.1f} / 10")
    print(f"Council wins          : {len(council_wins)}")
    print(f"Giant wins            : {len(giant_wins)}")
    print(f"Ties (within 1 point) : {len(ties)}")

    if council_wins:
        print("\nQuestions where Council won:")
        for r in council_wins:
            print(f"  - {r['question']} (Council {r['council_score']} vs Giant {r['giant_score']})")

    if giant_wins:
        print("\nQuestions where Giant won:")
        for r in giant_wins:
            print(f"  - {r['question']} (Giant {r['giant_score']} vs Council {r['council_score']})")

    overall = "COUNCIL" if council_avg > giant_avg else "GIANT" if giant_avg > council_avg else "TIE"
    print(f"\nOverall winner: {overall}")
    print("===================\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("   Council vs Giant  --  Local LLM Benchmark")
    print("=" * 60)
    print(f"\nCouncil models : {', '.join(COUNCIL_MODELS)}")
    print(f"Giant model    : {GIANT_MODEL}")
    print(f"Questions      : {len(QUESTIONS)}")
    print()

    # Quick connectivity check
    try:
        requests.get("http://localhost:11434/api/tags", timeout=5)
    except requests.ConnectionError:
        print("ERROR: Cannot connect to Ollama at http://localhost:11434")
        print("Make sure Ollama is running (ollama serve).")
        sys.exit(1)

    results = []
    total = len(QUESTIONS)
    start_all = time.time()

    for idx, item in enumerate(QUESTIONS, 1):
        question = item["question"]
        ground_truth = item["ground_truth"]

        print(f"\n[{idx}/{total}] {question}")
        print(f"  Ground truth: {ground_truth}")
        q_start = time.time()

        # --- Step 1 & 2 & 3: Council mode (parallel query + purge + synthesis) ---
        print("  Running council mode...")
        council_answers, council_synthesized = asyncio.run(run_council(question))
        for model, ans in council_answers.items():
            short = ans[:80].replace("\n", " ")
            print(f"    {model}: {short}...")
        short_synth = council_synthesized[:80].replace("\n", " ")
        print(f"    Synthesized: {short_synth}...")

        # --- Step 4: Giant mode (32B already loaded from synthesis) ---
        print("  Running giant mode...")
        giant_answer = run_giant(question)
        short_giant = giant_answer[:80].replace("\n", " ")
        print(f"    Giant: {short_giant}...")

        # --- Step 5: Judge both answers ---
        print("  Judging answers...")
        council_score = judge_answer(ground_truth, council_synthesized)
        giant_score = judge_answer(ground_truth, giant_answer)
        print(f"    Council score: {council_score}/10  |  Giant score: {giant_score}/10")

        elapsed = time.time() - q_start
        print(f"  Time: {elapsed:.1f}s")

        results.append({
            "question": question,
            "ground_truth": ground_truth,
            "council_individual_answers": council_answers,
            "council_answer": council_synthesized,
            "giant_answer": giant_answer,
            "council_score": council_score,
            "giant_score": giant_score,
            "time_seconds": round(elapsed, 1),
        })

    total_time = time.time() - start_all

    # --- Output ---
    print_table(results)
    print_summary(results)
    print(f"Total benchmark time: {total_time / 60:.1f} minutes")

    # Save full results to JSON
    output_path = "results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Full results saved to {output_path}")


if __name__ == "__main__":
    main()
