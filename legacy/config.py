"""Centralized configuration for Ollama models and endpoints."""

OLLAMA_BASE = "http://localhost:11434"
OLLAMA_GENERATE = f"{OLLAMA_BASE}/api/generate"
OLLAMA_TAGS = f"{OLLAMA_BASE}/api/tags"

# Council = small/tiny models that collaborate
COUNCIL_MODELS = ["qwen2.5:0.5b", "tinyllama", "gemma2:2b"]

# Giant = the single larger model that works alone (and acts as judge/synthesizer)
GIANT_MODEL = "phi3:mini"

# Timeout for a single generation call (seconds)
GENERATE_TIMEOUT = 300
