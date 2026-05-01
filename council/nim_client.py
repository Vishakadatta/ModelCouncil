"""NVIDIA NIM streaming client.

Implements the same (text, raw_dict) async generator interface as Ollama's
_stream_generate so orchestrator.py can route to either backend without any
logic changes in the council/judge pipeline.

NIM uses the OpenAI-compatible /v1/chat/completions endpoint with SSE.
keep_alive is an Ollama-only concept — it is NEVER sent to NIM.

raw_dict shape (matches what orchestrator/server expect from Ollama):
  {
    "response":   str,   # token text (empty on final frame)
    "done":       bool,  # True on the last frame only
    "eval_count": int,   # approximate tokens generated (set when done=True)
  }
"""

from __future__ import annotations

import json
import os
from typing import AsyncIterator

import httpx

DEFAULT_NIM_BASE = "https://integrate.api.nvidia.com/v1"
MAX_TOKENS = 1024            # default — fine for council members
JUDGE_MAX_TOKENS = 2048      # the regular judge needs more room for review + verdict
ULTIMATE_MAX_TOKENS = 3500   # the 405B should not be cut short — it's the final word
TEMPERATURE = 0.7

# Pool of council-tier western text models (4B–14B).
# Ordered largest→smallest so the strongest available is tried first.
# When the configured model returns 503 or 404, stream_generate
# automatically retries with the next available model.
# All models below have been verified live on NIM's free tier.
COUNCIL_FALLBACK_POOL: list[str] = [
    # 10B+ tier (preferred — better reasoning)
    "mistralai/ministral-14b-instruct-2512",       # 14B  · Mistral · France
    "google/gemma-3-12b-it",                       # 12B  · Google · USA
    "meta/llama-3.2-11b-vision-instruct",          # 11B  · Meta   · USA (multimodal but answers text)
    "nvidia/nvidia-nemotron-nano-9b-v2",           #  9B  · NVIDIA · USA
    # 7-8B fallback tier
    "meta/llama-3.1-8b-instruct",                  #  8B  · Meta   · USA
    "mistralai/mistral-7b-instruct-v0.3",          #  7B  · Mistral · France
    # Smaller fallback (last resort if all bigger models 503)
    "google/gemma-3-4b-it",                        #  4B  · Google · USA
    "meta/llama-3.2-3b-instruct",                  #  3B  · Meta   · USA
]

# Pool of judge-tier models (49B–123B). Ordered by capability/speed tradeoff.
# 405B is excluded by default — too slow on free tier (30s+ TTFT).
# All models below have been verified live on NIM's free tier.
JUDGE_FALLBACK_POOL: list[str] = [
    # 100B+ flagship tier
    "nvidia/nemotron-3-super-120b-a12b",           # 120B · NVIDIA  · USA · MoE, fast
    "mistralai/devstral-2-123b-instruct-2512",     # 123B · Mistral · France
    "mistralai/mistral-small-4-119b-2603",         # 119B · Mistral · France
    # 70B tier
    "meta/llama-3.3-70b-instruct",                 #  70B · Meta    · USA
    "meta/llama-3.1-70b-instruct",                 #  70B · Meta    · USA
    # 49B tier (Nemotron Super)
    "nvidia/llama-3.3-nemotron-super-49b-v1.5",    #  49B · NVIDIA  · USA
    "nvidia/llama-3.3-nemotron-super-49b-v1",      #  49B · NVIDIA  · USA
]


async def stream_generate(
    model: str,
    prompt: str,
    api_key: str | None = None,
    base: str | None = None,
    fallback_pool: list[str] | None = None,
    max_tokens: int = MAX_TOKENS,
) -> AsyncIterator[tuple[str, dict]]:
    """
    Stream tokens from NVIDIA NIM for a single prompt.

    Yields (chunk_text, raw_dict) per token, matching Ollama's
    _stream_generate interface so orchestrator.py needs no logic changes.

    On 503 (free-tier queue full), automatically retries with the next
    model in fallback_pool (defaults to COUNCIL_FALLBACK_POOL).  The
    actual model used is emitted as raw_dict["model"] on the first chunk
    so the UI can show which fallback was used.

    Args:
        model:         NIM model ID, e.g. "meta/llama-3.1-8b-instruct"
        prompt:        The full prompt string.
        api_key:       NVIDIA API key. Reads NVIDIA_API_KEY from env if None.
        base:          NIM base URL. Reads NIM_BASE from env if None.
        fallback_pool: Ordered list of models to try after the primary.

    Raises:
        RuntimeError:  NVIDIA_API_KEY missing, or all fallback models exhausted.
    """
    api_key = api_key or os.environ.get("NVIDIA_API_KEY", "")
    base = (base or os.environ.get("NIM_BASE", DEFAULT_NIM_BASE)).rstrip("/")

    if not api_key:
        raise RuntimeError(
            "NVIDIA_API_KEY is not set. Run `make setup` and choose the NIM backend."
        )

    if fallback_pool is None:
        fallback_pool = COUNCIL_FALLBACK_POOL

    # Build candidate list: requested model first, then fallbacks (skip dupes)
    candidates = [model] + [m for m in fallback_pool if m != model]

    last_error: Exception | None = None

    for candidate in candidates:
        try:
            async for item in _stream_once(candidate, prompt, api_key, base, max_tokens):
                yield item
            return  # success
        except httpx.HTTPStatusError as e:
            if e.response.status_code in (400, 404, 503):
                # 400 = NIM marks the model "DEGRADED function cannot be invoked"
                #       — transient outage on NVIDIA's side
                # 404 = model not available on this tier
                # 503 = free-tier queue full
                last_error = e
                continue  # try next candidate
            raise  # other 4xx/5xx (401 auth, 429 rate-limit) — don't retry

    raise RuntimeError(
        f"All NIM fallback models returned 503 (queue full). "
        f"Tried: {candidates}. Last error: {last_error}"
    )


async def _stream_once(
    model: str,
    prompt: str,
    api_key: str,
    base: str,
    max_tokens: int = MAX_TOKENS,
) -> AsyncIterator[tuple[str, dict]]:
    """Single attempt to stream from one model — no retry logic."""
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
        "max_tokens": max_tokens,
        "temperature": TEMPERATURE,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    token_count = 0
    first_chunk = True
    server_model: str | None = None  # what NVIDIA's response says identified itself as

    async with httpx.AsyncClient(timeout=600.0) as client:
        async with client.stream(
            "POST",
            f"{base}/chat/completions",
            json=payload,
            headers=headers,
        ) as resp:
            resp.raise_for_status()

            async for raw_line in resp.aiter_lines():
                if not raw_line:
                    continue
                if not raw_line.startswith("data: "):
                    continue

                data_str = raw_line[6:]

                if data_str.strip() == "[DONE]":
                    yield "", {
                        "response": "",
                        "done": True,
                        "eval_count": token_count,
                        "model": model,
                        "server_model": server_model,
                    }
                    return

                try:
                    obj = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                # Capture NIM's self-reported model on first chunk that has it.
                # This is independent confirmation of which model handled the call.
                if server_model is None:
                    server_model = obj.get("model")

                choices = obj.get("choices", [])
                if not choices:
                    continue

                delta = choices[0].get("delta", {})
                text: str = delta.get("content") or ""
                finish_reason = choices[0].get("finish_reason")
                done = finish_reason is not None

                if text:
                    token_count += 1

                raw_dict = {
                    "response": text,
                    "done": done,
                    "eval_count": token_count if done else 0,
                    "model": model,                # what we requested
                    "server_model": server_model,  # what NIM says it actually is
                }

                # On first chunk, signal if a fallback was activated
                if first_chunk:
                    raw_dict["fallback_model"] = model
                    first_chunk = False

                yield text, raw_dict

                if done:
                    return


def validate_key(api_key: str) -> bool:
    """
    Synchronous check: hit GET /v1/models with the key.
    Returns True on 200, False on 401/403.
    Raises httpx.HTTPError on unexpected network failures.
    """
    r = httpx.get(
        f"{DEFAULT_NIM_BASE}/models",
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=15.0,
    )
    return r.status_code == 200
