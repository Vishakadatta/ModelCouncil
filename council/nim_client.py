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


async def stream_generate(
    model: str,
    prompt: str,
    api_key: str | None = None,
    base: str | None = None,
) -> AsyncIterator[tuple[str, dict]]:
    """
    Stream tokens from NVIDIA NIM for a single prompt.

    Yields (chunk_text, raw_dict) per token, matching Ollama's
    _stream_generate interface so orchestrator.py needs no logic changes.

    Args:
        model:   NIM model ID, e.g. "meta/llama-3.1-8b-instruct"
        prompt:  The full prompt string.
        api_key: NVIDIA API key. Reads NVIDIA_API_KEY from env if None.
        base:    NIM base URL. Reads NIM_BASE from env if None.

    Raises:
        RuntimeError:           NVIDIA_API_KEY missing.
        httpx.HTTPStatusError:  NIM returned 4xx/5xx.
        httpx.HTTPError:        Network-level failure.
    """
    api_key = api_key or os.environ.get("NVIDIA_API_KEY", "")
    base = (base or os.environ.get("NIM_BASE", DEFAULT_NIM_BASE)).rstrip("/")

    if not api_key:
        raise RuntimeError(
            "NVIDIA_API_KEY is not set. Run `make setup` and choose the NIM backend."
        )

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

                data_str = raw_line[6:]  # strip "data: " prefix

                # SSE stream terminator
                if data_str.strip() == "[DONE]":
                    yield "", {
                        "response": "",
                        "done": True,
                        "eval_count": token_count,
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
                }
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
