"""Approved model allowlist, NIM publisher map, and tier policy.

APPROVED_MODELS is the production allowlist for the Ollama backend.
TEST_ONLY_MODELS is a separate bucket for `make demo` and the test suite.

NIM_PUBLISHER_MAP and NIM_BLOCKED_PUBLISHERS are used by both:
  - setup/nim_discover.py  (discovery + filtering at setup time)
  - origin_for()           (display labels at runtime)
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

# ---------------------------------------------------------------------------
# NIM publisher registry — shared by nim_discover.py and origin_for()
# Keys are lowercase publisher prefixes from NIM model IDs ("publisher/name").
# ---------------------------------------------------------------------------

NIM_PUBLISHER_MAP: dict[str, tuple[str, str]] = {
    # North America
    "meta":             ("Meta",          "USA"),
    "nvidia":           ("NVIDIA",        "USA"),
    "microsoft":        ("Microsoft",     "USA"),
    "google":           ("Google",        "USA"),
    "cohere":           ("Cohere",        "Canada"),
    "writer":           ("Writer",        "USA"),
    "adept":            ("Adept",         "USA"),
    "snowflake":        ("Snowflake",     "USA"),
    "databricks":       ("Databricks",    "USA"),
    "togethercomputer": ("Together AI",   "USA"),
    "teknium":          ("Teknium",       "USA"),
    "nomic-ai":         ("Nomic AI",      "USA"),
    "allenai":          ("Allen AI",      "USA"),
    "salesforce":       ("Salesforce",    "USA"),
    "mosaic":           ("MosaicML",      "USA"),
    "mosaicml":         ("MosaicML",      "USA"),
    "ibm":              ("IBM",           "USA"),
    "eleutherai":       ("EleutherAI",    "USA"),
    "openai":           ("OpenAI",        "USA"),
    "abacusai":         ("Abacus.AI",     "USA"),
    "zyphra":           ("Zyphra",        "USA"),
    # Europe
    "mistralai":        ("Mistral AI",    "France"),
    "nv-mistralai":     ("Mistral AI",    "France"),
    "bigscience":       ("BigScience",    "France"),
    "stabilityai":      ("Stability AI",  "UK"),
    # Middle East / Asia-Pacific (non-China)
    "tiiuae":           ("TII",           "UAE"),
    "upstage":          ("Upstage",       "South Korea"),
    "llmjp":            ("LLM-jp",        "Japan"),
    "cyberagent":       ("CyberAgent",    "Japan"),
    "stockmark":        ("Stockmark",     "Japan"),
    "ai21labs":         ("AI21 Labs",     "Israel"),
    "aisingapore":      ("AI Singapore",  "Singapore"),
    "sarvamai":         ("Sarvam AI",     "India"),
    # Community / open-source (treated as neutral)
    "openchat":         ("OpenChat",      "Community"),
    "garage-baind":     ("Garage-bAInd",  "Community"),
    "bigcode":          ("BigCode",       "Community"),
}

# NIM publishers whose models MUST be rejected.
# This list encodes the policy: no Chinese-origin models.
NIM_BLOCKED_PUBLISHERS: set[str] = {
    # Alibaba / Qwen family
    "qwen", "alibaba", "alibabacloud",
    # Baidu
    "baidu",
    # ByteDance
    "bytedance",
    # Tencent
    "tencent",
    # DeepSeek
    "deepseek-ai", "deepseek",
    # 01.ai / Yi
    "01-ai", "01ai",
    # THUDM / Zhipu (Tsinghua spinoff)
    "thudm", "zhipu-ai", "zhipuai",
    # MiniMax / Moonshot / Kimi  (NIM uses no-hyphen forms: minimaxai, moonshotai)
    "minimax-ai", "minimaxai", "minimax", "moonshot-ai", "moonshotai", "moonshot", "kimi",
    # Baichuan
    "baichuan-inc", "baichuan",
    # InternLM / Shanghai AI Lab
    "internlm", "shanghaiailab",
    # SenseTime / Megvii
    "senseauto", "sensetime", "megvii",
    # ModelBest / IDEA Research
    "modelbest", "idea-ccnl", "idea-research",
    # University-affiliated Chinese labs
    "pkuslam",    # Peking University
    "thu-coai",   # Tsinghua COAI
    "fudan-nlp",  # Fudan University
    # Zhipu AI / GLM family  (NIM publisher prefix: z-ai)
    "z-ai", "zhipu-ai", "zhipuai", "glm",
    # Step AI (Chinese)
    "stepfun-ai", "stepfun",
    # Beijing Academy of Artificial Intelligence
    "baai",
    # Other Chinese AI companies
    "infini-ai", "openbmb",
}


class ModelPolicyError(ValueError):
    """Raised when a non-approved model is used in production context."""


def approved_tags(role: str | None = None) -> list[str]:
    return [m.tag for m in APPROVED_MODELS if role is None or m.role == role]


def origin_for(tag: str) -> str:
    """
    Return a human-readable origin string for a model tag.

    Checks the static Ollama allowlist first, then falls back to
    NIM publisher prefix inference for tags like "meta/llama-3.1-8b-instruct".
    """
    # Static Ollama allowlist (exact match)
    for m in APPROVED_MODELS:
        if m.tag == tag:
            return m.origin
    # NIM publisher prefix inference
    return _nim_origin_for(tag)


def _nim_origin_for(tag: str) -> str:
    """Infer origin from NIM model ID publisher prefix."""
    publisher = tag.split("/", 1)[0].lower() if "/" in tag else tag.split(":")[0].lower()
    entry = NIM_PUBLISHER_MAP.get(publisher)
    if entry:
        company, country = entry
        return f"{company}, {country}"
    return "unknown"


def vram_for(tag: str) -> int:
    for m in APPROVED_MODELS:
        if m.tag == tag:
            return m.vram_gb
    raise ModelPolicyError(f"Unknown model tag: {tag}")


def assert_production_allowed(tag: str, role: str) -> None:
    """
    Raise if tag is not on the Ollama approved list for the given role.
    This check is Ollama-specific. NIM models bypass it because
    nim_discover.py enforces its own origin + size policy at setup time.
    """
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
