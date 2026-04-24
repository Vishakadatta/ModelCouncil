"""Pipeline tests with a fake transport — no GPU, no network.

What we verify:
  - keep_alive=0 is present in EVERY council generate call (Ollama path).
  - Council enforces vendor diversity (no duplicate models).
  - Production policy rejects test-only / unknown models (Ollama path).
  - End-to-end session JSON matches the documented schema.
  - NIM discovery correctly rejects Chinese-origin models.
  - NIM discovery correctly classifies model sizes.
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
            "url":        str(request.url),
            "model":      body.get("model"),
            "stream":     body.get("stream"),
            "keep_alive": body.get("keep_alive"),
            "prompt":     body.get("prompt", "")[:80],
        })
        if body.get("stream"):
            chunks = ["Hello ", "from ", body["model"], "."]
            lines  = [
                json.dumps({"response": c, "done": False}) for c in chunks
            ] + [json.dumps({"response": "", "done": True, "eval_count": 4})]
            return httpx.Response(200, text="\n".join(lines) + "\n")
        return httpx.Response(
            200,
            json={
                "response":   f"answer from {body['model']}",
                "eval_count": 7,
                "done":       True,
            },
        )


@pytest.fixture
def fake_ollama(monkeypatch):
    fake      = FakeOllama()
    transport = httpx.MockTransport(fake.handler)
    real_async_client = httpx.AsyncClient

    def patched(*args, **kwargs):
        kwargs["transport"] = transport
        return real_async_client(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", patched)
    return fake


# ---------------------------------------------------------------------------
# Policy tests (Ollama allowlist)
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
    assert_production_allowed("gemma2:27b",  "judge")


# ---------------------------------------------------------------------------
# Pipeline tests (Ollama path)
# ---------------------------------------------------------------------------

def test_keep_alive_zero_in_every_council_call(fake_ollama):
    """Non-negotiable invariant: every council generate uses keep_alive=0."""
    orch = Orchestrator(
        council_models=["llama3.1:8b", "mistral:7b", "gemma2:9b"],
        judge_model="gemma2:27b",
    )
    asyncio.run(orch.run_council("ping"))

    council_calls = [
        c for c in fake_ollama.calls if c["model"] in orch.council_models
    ]
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

    out  = asyncio.run(go())
    data = json.loads(out.read_text())

    assert data["plan"]   == "balanced"
    assert data["prompt"] == "What is gravity?"
    assert len(data["council"]) == 3
    for turn in data["council"]:
        for k in ("model", "origin", "response", "latency_seconds", "tokens_generated"):
            assert k in turn, f"council turn missing field {k}"
    assert data["judge"]["model"]  == "gemma2:27b"
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


# ---------------------------------------------------------------------------
# NIM discovery policy tests — no network needed
# ---------------------------------------------------------------------------

def test_nim_blocks_chinese_publishers():
    """is_blocked_origin must return True for every Chinese-origin publisher."""
    from setup.nim_discover import is_blocked_origin

    blocked = [
        "qwen",          # Alibaba
        "alibaba",
        "deepseek-ai",   # DeepSeek
        "deepseek",
        "baidu",
        "bytedance",
        "tencent",
        "01-ai",
        "thudm",         # Tsinghua spinoff
        "zhipu-ai",
        "minimax-ai",
        "moonshot-ai",
        "baichuan-inc",
        "internlm",
        "sensetime",
    ]
    for pub in blocked:
        assert is_blocked_origin(pub), f"Expected {pub!r} to be blocked"


def test_nim_allows_western_publishers():
    """is_blocked_origin must return False for approved western publishers."""
    from setup.nim_discover import is_blocked_origin

    allowed = ["meta", "mistralai", "google", "microsoft", "cohere", "nvidia"]
    for pub in allowed:
        assert not is_blocked_origin(pub), f"Expected {pub!r} to be allowed"


def test_nim_fetch_filters_chinese_models(monkeypatch):
    """
    A mocked /v1/models response containing a Qwen (Alibaba) model must
    produce zero council entries for that model after fetch_nim_models().
    """
    from setup.nim_discover import fetch_nim_models

    fake_catalogue = {
        "data": [
            {"id": "meta/llama-3.1-8b-instruct"},          # council — USA
            {"id": "qwen/qwen2-7b-instruct"},               # BLOCKED — China
            {"id": "mistralai/mistral-7b-instruct-v0.3"},   # council — France
            {"id": "deepseek-ai/deepseek-r1-distill-70b"},  # BLOCKED — China
            {"id": "nvidia/llama-3.1-nemotron-70b-instruct"},  # judge — USA
        ]
    }

    def mock_get(url, **kwargs):
        # httpx.Response needs an attached request for raise_for_status() to work.
        req = httpx.Request("GET", url)
        return httpx.Response(200, json=fake_catalogue, request=req)

    monkeypatch.setattr(httpx, "get", mock_get)

    models     = fetch_nim_models("test-key-doesnt-matter")
    model_ids  = [m.model_id for m in models]

    # Chinese models must be absent
    assert "qwen/qwen2-7b-instruct"              not in model_ids, "Qwen not filtered"
    assert "deepseek-ai/deepseek-r1-distill-70b" not in model_ids, "DeepSeek not filtered"

    # Western models must be present
    assert "meta/llama-3.1-8b-instruct"                   in model_ids
    assert "mistralai/mistral-7b-instruct-v0.3"           in model_ids
    assert "nvidia/llama-3.1-nemotron-70b-instruct"       in model_ids


def test_nim_param_extraction():
    """_extract_param_b must correctly parse parameter counts from model names."""
    from setup.nim_discover import _extract_param_b

    cases = [
        ("meta/llama-3.1-8b-instruct",            8.0),
        ("mistralai/mistral-7b-instruct-v0.3",     7.0),
        ("nvidia/llama-3.1-nemotron-70b-instruct", 70.0),
        ("microsoft/phi-3-mini-3.8b",              3.8),
        ("mistralai/mixtral-8x7b-instruct-v0.1",   56.0),  # MoE
        ("meta/llama-3.1-405b-instruct",           405.0),
    ]
    for model_id, expected in cases:
        got = _extract_param_b(model_id)
        assert got == expected, f"{model_id}: expected {expected}, got {got}"


def test_nim_role_classification():
    """Models <15B → council, ≥30B → judge, 15–29B → None (gap)."""
    from setup.nim_discover import _classify_role

    assert _classify_role(7.0)   == "council"
    assert _classify_role(8.0)   == "council"
    assert _classify_role(14.9)  == "council"
    assert _classify_role(15.0)  is None    # ambiguous gap
    assert _classify_role(27.0)  is None    # ambiguous gap
    assert _classify_role(30.0)  == "judge"
    assert _classify_role(70.0)  == "judge"
    assert _classify_role(405.0) == "judge"
