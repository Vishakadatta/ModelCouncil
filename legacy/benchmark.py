#!/usr/bin/env python3
"""
CLI Benchmark runner - uses the modular council package.
This replaces the original monolithic council_vs_giant.py.

Run with: python3 benchmark.py
"""

import json
import sys
import time

from council.config import COUNCIL_MODELS, GIANT_MODEL
from council.ollama import is_ollama_running
from council.engine import run_council_mode, query_single
from council.ollama import generate
from council.judge import score_answer
from council.questions import BENCHMARK_QUESTIONS


def print_table(results):
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
        cs, gs = r["council_score"], r["giant_score"]
        diff = cs - gs
        winner = "Council" if diff > 1 else "Giant" if diff < -1 else "Tie"
        print(f"{i:<4} {q:<{q_width}} {cs:>8} {gs:>8} {winner:>10}")
    print(sep)


def print_summary(results):
    c_scores = [r["council_score"] for r in results]
    g_scores = [r["giant_score"] for r in results]
    c_avg = sum(c_scores) / len(c_scores)
    g_avg = sum(g_scores) / len(g_scores)
    c_wins = [r for r in results if r["council_score"] - r["giant_score"] > 1]
    g_wins = [r for r in results if r["giant_score"] - r["council_score"] > 1]
    ties = [r for r in results if abs(r["council_score"] - r["giant_score"]) <= 1]

    print("\n===== SUMMARY =====")
    print(f"Council average score : {c_avg:.1f} / 10")
    print(f"Giant average score   : {g_avg:.1f} / 10")
    print(f"Council wins          : {len(c_wins)}")
    print(f"Giant wins            : {len(g_wins)}")
    print(f"Ties (within 1 point) : {len(ties)}")

    if c_wins:
        print("\nQuestions where Council won:")
        for r in c_wins:
            print(f"  - {r['question']} (Council {r['council_score']} vs Giant {r['giant_score']})")
    if g_wins:
        print("\nQuestions where Giant won:")
        for r in g_wins:
            print(f"  - {r['question']} (Giant {r['giant_score']} vs Council {r['council_score']})")

    overall = "COUNCIL" if c_avg > g_avg else "GIANT" if g_avg > c_avg else "TIE"
    print(f"\nOverall winner: {overall}")
    print("===================\n")


def main():
    print("=" * 60)
    print("   Council vs Giant  --  Local LLM Benchmark")
    print("=" * 60)
    print(f"\nCouncil : {', '.join(COUNCIL_MODELS)}")
    print(f"Giant   : {GIANT_MODEL}")
    print(f"Questions: {len(BENCHMARK_QUESTIONS)}\n")

    if not is_ollama_running():
        print("ERROR: Ollama not reachable. Run: ollama serve")
        sys.exit(1)

    results = []
    total = len(BENCHMARK_QUESTIONS)
    start_all = time.time()

    for idx, item in enumerate(BENCHMARK_QUESTIONS, 1):
        question, gt = item["question"], item["ground_truth"]
        print(f"\n[{idx}/{total}] {question}")
        print(f"  Ground truth: {gt}")
        t0 = time.time()

        print("  Council mode...")
        council = run_council_mode(question)
        for m, a in council["individual"].items():
            print(f"    {m}: {a[:80].replace(chr(10), ' ')}...")
        print(f"    Judgment: {council['giant_judgment'][:80].replace(chr(10), ' ')}...")

        print("  Giant mode...")
        giant_answer = generate(GIANT_MODEL, question)
        print(f"    Giant: {giant_answer[:80].replace(chr(10), ' ')}...")

        print("  Scoring...")
        cs = score_answer(gt, council["giant_judgment"])
        gs = score_answer(gt, giant_answer)
        print(f"    Council: {cs}/10  |  Giant: {gs}/10")

        elapsed = round(time.time() - t0, 1)
        print(f"  Time: {elapsed}s")

        results.append({
            "question": question,
            "ground_truth": gt,
            "council_individual": council["individual"],
            "council_answer": council["giant_judgment"],
            "giant_answer": giant_answer,
            "council_score": cs,
            "giant_score": gs,
            "time_seconds": elapsed,
        })

    total_time = time.time() - start_all
    print_table(results)
    print_summary(results)
    print(f"Total time: {total_time / 60:.1f} minutes")

    with open("results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Results saved to results.json")


if __name__ == "__main__":
    main()
