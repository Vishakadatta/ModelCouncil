"""Core engine: council mode, giant mode, single-model queries, and cross-judging."""

import asyncio
from .config import COUNCIL_MODELS, GIANT_MODEL
from .ollama import generate, unload, async_generate


# ---------------------------------------------------------------------------
# Basic queries
# ---------------------------------------------------------------------------

def query_single(model: str, question: str) -> str:
    """Query one model directly and return its answer."""
    return generate(model, question)


async def _query_council_parallel(question: str) -> dict[str, str]:
    """Query all council models in parallel, return {model: answer}."""
    tasks = [async_generate(m, question) for m in COUNCIL_MODELS]
    results = await asyncio.gather(*tasks)
    return {model: answer for model, answer in results}


async def _council_judge_parallel(giant_answer: str, question: str) -> dict[str, str]:
    """Each council model judges the giant's answer in parallel."""
    prompt = (
        f"You are reviewing another AI's answer for accuracy.\n\n"
        f"Question: {question}\n\n"
        f"Answer to review:\n{giant_answer}\n\n"
        f"Judge this answer: Is it correct or wrong? Explain briefly, "
        f"then give your own best answer to the question."
    )
    tasks = [async_generate(m, prompt) for m in COUNCIL_MODELS]
    results = await asyncio.gather(*tasks)
    return {model: judgment for model, judgment in results}


# ---------------------------------------------------------------------------
# Giant judging council
# ---------------------------------------------------------------------------

def giant_judge_council(question: str, council_answers: dict[str, str]) -> str:
    """
    Giant model reads all council answers, judges each one,
    says which are correct/wrong, and gives its own final answer.
    """
    prompt = (
        f"You are a senior judge reviewing answers from three smaller AI models.\n\n"
        f"Question: {question}\n\n"
    )
    for model, answer in council_answers.items():
        prompt += f"--- {model} ---\n{answer}\n\n"
    prompt += (
        "For each model's response above:\n"
        "1. State whether it is CORRECT or WRONG (and why, briefly)\n"
        "2. Then give your own best, most accurate answer to the question."
    )
    return generate(GIANT_MODEL, prompt)


# ---------------------------------------------------------------------------
# Mode: Council Only
#   - 3 tiny models answer
#   - Giant judges their answers and gives verdict
# ---------------------------------------------------------------------------

def run_council_mode(question: str) -> dict:
    individual = asyncio.run(_query_council_parallel(question))

    for model in COUNCIL_MODELS:
        unload(model)

    judgment = giant_judge_council(question, individual)

    return {
        "individual": individual,
        "giant_judgment": judgment,
    }


# ---------------------------------------------------------------------------
# Mode: Giant Only
#   - Giant answers alone
#   - 3 tiny models each judge the giant's answer
# ---------------------------------------------------------------------------

def run_giant_mode(question: str) -> dict:
    giant_answer = generate(GIANT_MODEL, question)

    # Unload giant before loading tiny models
    unload(GIANT_MODEL)

    council_judgments = asyncio.run(_council_judge_parallel(giant_answer, question))

    # Unload tiny models after judging
    for model in COUNCIL_MODELS:
        unload(model)

    return {
        "giant_answer": giant_answer,
        "council_judgments": council_judgments,
    }


# ---------------------------------------------------------------------------
# Mode: All
#   - 3 tiny models answer
#   - Giant answers alone
#   - Giant judges the council's answers (verdict + its own best answer)
# ---------------------------------------------------------------------------

def run_all_mode(question: str) -> dict:
    # Step 1: council answers in parallel
    individual = asyncio.run(_query_council_parallel(question))

    # Step 2: unload tiny models
    for model in COUNCIL_MODELS:
        unload(model)

    # Step 3: giant answers the question directly
    giant_answer = generate(GIANT_MODEL, question)

    # Step 4: giant judges the council answers
    judgment = giant_judge_council(question, individual)

    return {
        "council_individual": individual,
        "giant_answer": giant_answer,
        "giant_judgment": judgment,
    }
