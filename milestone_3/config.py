"""Shared configuration helpers for milestone 3 components.

Keep secrets in environment variables; do not commit real keys.
"""

from dataclasses import dataclass
import os
from typing import Optional

try:  # pragma: no cover - optional dependency
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None

if load_dotenv:
    # Auto-load .env if present to avoid hardcoding keys in code.
    load_dotenv()


def _infer_default_base() -> Optional[str]:
    """Prefer OpenRouter base if its key is set; otherwise respect env overrides."""
    if os.getenv("OPENROUTER_API_KEY"):
        return os.getenv("OPENROUTER_BASE", "https://openrouter.ai/api/v1")
    return os.getenv("OPENAI_BASE_URL")


@dataclass
class LLMConfig:
    """LLM provider settings - uses OpenRouter for all models."""

    provider: str = os.getenv("LLM_PROVIDER", "openrouter")
    model: str = os.getenv("LLM_MODEL", "google/gemini-2.5-flash-preview")
    api_key: Optional[str] = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
    base_url: Optional[str] = _infer_default_base()

    def require_api_key(self) -> str:
        if not self.api_key:
            raise RuntimeError(
                "No API key found. Set OPENROUTER_API_KEY in your environment."
            )
        return self.api_key


def load_llm_config() -> LLMConfig:
    """Return LLM config populated from environment variables."""
    return LLMConfig()


