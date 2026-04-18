"""Approved model allowlist and tier policy.

APPROVED_MODELS is the production allowlist. Anything outside it must be
rejected in production. TEST_ONLY_MODELS is a separate bucket for `make demo`
and the test suite — never permitted in production config.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    tag: str
    role: str          # "council" or "judge"
    origin: str
    vram_gb: int


APPROVED_MODELS: list[ModelSpec] = [
    ModelSpec("llama3.1:8b",    "council", "Meta, USA",          5),
    ModelSpec("mistral:7b",     "council", "Mistral AI, France", 5),
    ModelSpec("gemma2:9b",      "council", "Google, USA",        6),
    ModelSpec("phi3:mini",      "council", "Microsoft, USA",     3),
    ModelSpec("phi3:medium",    "council", "Microsoft, USA",     9),
    ModelSpec("command-r",      "council", "Cohere, Canada",     5),
    ModelSpec("gemma2:27b",     "judge",   "Google, USA",       18),
    ModelSpec("command-r-plus", "judge",   "Cohere, Canada",    22),
    ModelSpec("mistral:7b",     "judge",   "Mistral AI, France", 5),
    ModelSpec("llama3.1:8b",    "judge",   "Meta, USA",          5),
]

TEST_ONLY_MODELS: list[str] = [
    "tinyllama",
    "phi3:mini",
    "gemma2:2b",
    "qwen:0.5b",
]


class ModelPolicyError(ValueError):
    """Raised when a non-approved model is used in production context."""


def approved_tags(role: str | None = None) -> list[str]:
    return [m.tag for m in APPROVED_MODELS if role is None or m.role == role]


def origin_for(tag: str) -> str:
    for m in APPROVED_MODELS:
        if m.tag == tag:
            return m.origin
    return "unknown"


def vram_for(tag: str) -> int:
    for m in APPROVED_MODELS:
        if m.tag == tag:
            return m.vram_gb
    raise ModelPolicyError(f"Unknown model tag: {tag}")


def assert_production_allowed(tag: str, role: str) -> None:
    """Raise if tag is not on the approved list for the given role."""
    if tag in TEST_ONLY_MODELS and tag not in approved_tags(role):
        raise ModelPolicyError(
            f"Model '{tag}' is test-only and cannot be used in production "
            f"mode. Run `make demo` instead."
        )
    if tag not in approved_tags(role):
        raise ModelPolicyError(
            f"Model '{tag}' is not on the APPROVED_MODELS list for role "
            f"'{role}'. Approved {role} models: {approved_tags(role)}"
        )


def assert_council_diverse(tags: list[str]) -> None:
    """Council must use distinct models — no duplicates."""
    if len(tags) != len(set(tags)):
        raise ModelPolicyError(
            f"Council models must be diverse — duplicates not allowed: {tags}"
        )
    if len(tags) < 2:
        raise ModelPolicyError("Council requires at least 2 models.")
