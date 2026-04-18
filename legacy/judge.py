"""Scoring / judging logic using the giant model as evaluator."""

from .config import GIANT_MODEL
from .ollama import generate


def score_answer(ground_truth: str, answer: str) -> int:
    """
    Use the giant model to score an answer 0-10 for factual accuracy.
    Returns parsed integer score, or 5 on parse failure.
    """
    prompt = (
        f"Given the correct answer is {ground_truth}, score the following response "
        f"from 0 to 10 for factual accuracy only. Respond with just a number. "
        f"Response: {answer}"
    )
    raw = generate(GIANT_MODEL, prompt)
    return _parse_score(raw)


def _parse_score(raw: str) -> int:
    """Extract first valid 0-10 integer from model output."""
    for token in raw.split():
        cleaned = token.strip(".,;:!?/()[]**")
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
    return 5
