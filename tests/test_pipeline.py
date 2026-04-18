"""Pipeline tests with a fake Ollama transport — no GPU, no network.

What we verify:
  - keep_alive=0 is present in EVERY council generate call.
  - Council enforces vendor diversity (no duplicate models).
  - Production policy rejects test-only / unknown models.
  - End-to-end session JSON matches the documented schema.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import httpx
import pytest

from council.models import (
    ModelPolicyError,
    assert_council_diverse,
    assert_production_allowed,
)
from council.orchestrator import COUNCIL_KEEP_ALIVE, Orchestrator


# ---------------------------------------------------------------------------
# Fake Ollama transport
# ---------------------------------------------------------------------------

class FakeOllama:
    """Records every request; replies with deterministic Ollama-shaped JSON."""

    def __init__(self):
        self.calls: list[dict] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode())
        self.calls.append({
            "url": str(request.url),
            "model": body.get("model"),
            "stream": body.get("stream"),
            "keep_alive": body.get("keep_alive"),
            "prompt": body.get("prompt", "")[:80],
        })
        if body.get("stream"):
            # Streaming: NDJSON of chunks then a done frame.
            chunks = ["Hello ", "from ", body["model"], "."]
            lines = [
                json.dumps({"response": c, "done": False}) for c in chunks
            ] + [json.dumps({"response": "", "done": True, "eval_count": 4})]
            return httpx.Response(200, text="\n".join(lines) + "\n")
        return httpx.Response(
            200,
            json={
                "response": f"answer from {body['model']}",
                "eval_count": 7,
                "done": True,
            },
        )


@pytest.fixture
def fake_ollama(monkeypatch):
    fake = FakeOllama()
    transport = httpx.MockTransport(fake.handler)

    real_async_client = httpx.AsyncClient

    def patched(*args, **kwargs):
        kwargs["transport"] = transport
        return real_async_client(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", patched)
    return fake


# ---------------------------------------------------------------------------
# Policy tests
# ---------------------------------------------------------------------------

def test_council_diverse_rejects_duplicates():
    with pytest.raises(ModelPolicyError):
        assert_council_diverse(["mistral:7b", "mistral:7b"])


def test_test_only_model_rejected_in_production():
    with pytest.raises(ModelPolicyError, match="test-only"):
        assert_production_allowed("tinyllama", "council")


def test_unknown_model_rejected():
    with pytest.raises(ModelPolicyError):
        assert_production_allowed("not-a-real-model", "council")


def test_approved_model_accepted():
    assert_production_allowed("llama3.1:8b", "council")
    assert_production_allowed("gemma2:27b", "judge")


# ---------------------------------------------------------------------------
# Pipeline tests
# ---------------------------------------------------------------------------

def test_keep_alive_zero_in_every_council_call(fake_ollama):
    """The non-negotiable invariant: every council generate uses keep_alive=0."""
    orch = Orchestrator(
        council_models=["llama3.1:8b", "mistral:7b", "gemma2:9b"],
        judge_model="gemma2:27b",
    )
    asyncio.run(orch.run_council("ping"))

    council_calls = [c for c in fake_ollama.calls if c["model"] in orch.council_models]
    assert len(council_calls) == 3
    for call in council_calls:
        assert call["keep_alive"] == COUNCIL_KEEP_ALIVE, (
            f"keep_alive must be 0 on every council call, got {call}"
        )
        assert call["stream"] is False


def test_judge_call_does_not_use_keep_alive_zero(fake_ollama):
    """Judge persists in VRAM during synthesis — uses default keep_alive."""
    orch = Orchestrator(
        council_models=["llama3.1:8b", "mistral:7b"],
        judge_model="gemma2:27b",
    )
    council_turns = asyncio.run(orch.run_council("ping"))
    asyncio.run(orch.run_judge_stream("ping", council_turns))

    judge_calls = [c for c in fake_ollama.calls if c["model"] == "gemma2:27b"]
    assert judge_calls, "expected at least one judge call"
    assert judge_calls[0]["stream"] is True
    assert judge_calls[0]["keep_alive"] != 0


def test_full_session_writes_valid_json(fake_ollama, tmp_path: Path):
    from council.orchestrator import Session
    from datetime import datetime

    orch = Orchestrator(
        council_models=["llama3.1:8b", "mistral:7b", "gemma2:9b"],
        judge_model="gemma2:27b",
        plan_name="balanced",
    )

    async def go():
        turns = await orch.run_council("What is gravity?")
        judge = await orch.run_judge_stream("What is gravity?", turns)
        session = Session(
            timestamp=datetime.now().isoformat(timespec="seconds"),
            prompt="What is gravity?",
            plan="balanced",
            council=turns,
            judge=judge,
            total_seconds=1.0,
        )
        out = tmp_path / "session.json"
        out.write_text(json.dumps(session.to_dict(), indent=2))
        return out

    out = asyncio.run(go())
    data = json.loads(out.read_text())

    assert data["plan"] == "balanced"
    assert data["prompt"] == "What is gravity?"
    assert len(data["council"]) == 3
    for turn in data["council"]:
        for k in ("model", "origin", "response", "latency_seconds", "tokens_generated"):
            assert k in turn, f"council turn missing field {k}"
    assert data["judge"]["model"] == "gemma2:27b"
    assert data["judge"]["origin"] == "Google, USA"
    assert "response" in data["judge"]
    assert isinstance(data["total_seconds"], (int, float))


def test_orchestrator_rejects_duplicate_council_in_production():
    with pytest.raises(ModelPolicyError):
        Orchestrator(
            council_models=["mistral:7b", "mistral:7b"],
            judge_model="gemma2:27b",
        )


def test_orchestrator_allows_test_models_when_production_false():
    """Demo mode disables the policy check so tiny models can run."""
    orch = Orchestrator(
        council_models=["tinyllama", "phi3:mini"],
        judge_model="phi3:mini",
        production=False,
    )
    assert orch.council_models == ["tinyllama", "phi3:mini"]
