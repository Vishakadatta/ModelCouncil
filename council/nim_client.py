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
MAX_TOKENS = 1024
TEMPERATURE = 0.7

# Pool of small western text models (<30B).
# When the configured model returns 503 (queue full on free tier),
# stream_generate automatically retries with the next available model.
# Ordered roughly smallest→largest so cheaper models are tried first.
COUNCIL_FALLBACK_POOL: list[str] = [
    "meta/llama-3.2-3b-instruct",
    "google/gemma-3-4b-it",
    "microsoft/phi-4-mini-instruct",
    "meta/llama-3.2-1b-instruct",
    "ibm/granite-3.0-3b-a800m-instruct",
    "meta/llama-3.1-8b-instruct",
    "mistralai/mistral-7b-instruct-v0.3",
    "ibm/granite-3.0-8b-instruct",
    "nvidia/llama-3.1-nemotron-nano-8b-v1",
    "google/gemma-3-12b-it",
    "nv-mistralai/mistral-nemo-12b-instruct",
    "mistralai/ministral-14b-instruct-2512",
    "microsoft/phi-3.5-moe-instruct",
]

JUDGE_FALLBACK_POOL: list[str] = [
    "nvidia/llama-3.1-nemotron-70b-instruct",
    "meta/llama-3.1-70b-instruct",
    "meta/llama-3.3-70b-instruct",
    "mistralai/mistral-large",
    "nvidia/llama-3.1-nemotron-51b-instruct",
]


async def stream_generate(
    model: str,
    prompt: str,
    api_key: str | None = None,
    base: str | None = None,
    fallback_pool: list[str] | None = None,
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
            async for item in _stream_once(candidate, prompt, api_key, base):
                yield item
            return  # success
        except httpx.HTTPStatusError as e:
            if e.response.status_code in (404, 503):
                # 404 = model not available on this tier
                # 503 = free-tier queue full
                last_error = e
                continue  # try next candidate
            raise  # other 4xx/5xx — don't retry

    raise RuntimeError(
        f"All NIM fallback models returned 503 (queue full). "
        f"Tried: {candidates}. Last error: {last_error}"
    )


async def _stream_once(
    model: str,
    prompt: str,
    api_key: str,
    base: str,
) -> AsyncIterator[tuple[str, dict]]:
    """Single attempt to stream from one model — no retry logic."""
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    token_count = 0
    first_chunk = True

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
                    }
                    return

                try:
                    obj = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

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
                    "model": model,  # actual model used (may differ from requested)
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
