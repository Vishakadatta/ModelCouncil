"""Low-level Ollama HTTP helpers. Every model interaction goes through here."""

import asyncio
import requests
from .config import OLLAMA_GENERATE, OLLAMA_TAGS, GENERATE_TIMEOUT


def is_ollama_running() -> bool:
    """Return True if Ollama is reachable."""
    try:
        requests.get(OLLAMA_TAGS, timeout=5)
        return True
    except requests.ConnectionError:
        return False


def list_models() -> list[str]:
    """Return names of all locally available models."""
    resp = requests.get(OLLAMA_TAGS, timeout=10)
    resp.raise_for_status()
    return [m["name"] for m in resp.json().get("models", [])]


def generate(model: str, prompt: str) -> str:
    """Send a prompt to a model and return the full response text."""
    payload = {"model": model, "prompt": prompt, "stream": False}
    resp = requests.post(OLLAMA_GENERATE, json=payload, timeout=GENERATE_TIMEOUT)
    resp.raise_for_status()
    return resp.json().get("response", "").strip()


def unload(model: str) -> None:
    """Free VRAM by setting keep_alive=0 for a model."""
    payload = {"model": model, "prompt": "", "stream": False, "keep_alive": 0}
    try:
        requests.post(OLLAMA_GENERATE, json=payload, timeout=60)
    except Exception:
        pass


async def async_generate(model: str, prompt: str) -> tuple[str, str]:
    """Run generate() in a thread pool for concurrent model queries."""
    loop = asyncio.get_event_loop()
    response = await loop.run_in_executor(None, generate, model, prompt)
    return model, response
