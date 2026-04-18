#!/usr/bin/env python3
"""Entrypoint: ask the council a question.

Usage:
  python council.py "your question here"
  python council.py --env .env.demo "your question"
  python council.py            # interactive prompt
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path

from council.models import ModelPolicyError
from council.orchestrator import Orchestrator, Session, load_env_config
from council.output import (
    banner_council_start,
    banner_judge_loading,
    policy_error,
    render_council_turn,
    saved,
    stream_judge,
)

RESULTS_DIR = Path("results")


def main() -> int:
    parser = argparse.ArgumentParser(description="Ask the model council.")
    parser.add_argument("prompt", nargs="*", help="The question to ask.")
    parser.add_argument("--env", default=".env", help="Path to env file.")
    parser.add_argument(
        "--plan-name", default="custom", help="Plan label saved with the session."
    )
    args = parser.parse_args()

    if args.env != ".env":
        _load_env_into_process(args.env)

    try:
        council, judge, base = load_env_config()
    except RuntimeError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    prompt_text = " ".join(args.prompt).strip() or input("Your question: ").strip()
    if not prompt_text:
        print("error: empty prompt", file=sys.stderr)
        return 2

    production = "demo" not in args.env.lower()

    try:
        orch = Orchestrator(
            council_models=council,
            judge_model=judge,
            ollama_base=base,
            plan_name=args.plan_name,
            production=production,
        )
    except ModelPolicyError as e:
        policy_error(str(e))
        return 3

    return asyncio.run(_run(orch, prompt_text))


async def _run(orch: Orchestrator, prompt_text: str) -> int:
    banner_council_start(orch.council_models)
    council_turns = await orch.run_council(prompt_text, on_done=render_council_turn)

    banner_judge_loading(orch.judge_model)
    with stream_judge(orch.judge_model) as append_chunk:
        judge_turn = await orch.run_judge_stream(
            prompt_text, council_turns, on_chunk=append_chunk
        )

    session = Session(
        timestamp=datetime.now().isoformat(timespec="seconds"),
        prompt=prompt_text,
        plan=orch.plan_name,
        council=council_turns,
        judge=judge_turn,
        total_seconds=sum(t.latency_seconds for t in council_turns) + judge_turn.latency_seconds,
    )
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    out.write_text(json.dumps(session.to_dict(), indent=2))
    saved(out)
    return 0


def _load_env_into_process(path: str) -> None:
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ[k.strip()] = v.strip()


if __name__ == "__main__":
    sys.exit(main())
