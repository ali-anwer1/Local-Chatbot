"""Abstract LLM backend interface and model-tag helper utilities (§4.4)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import Any


EMBEDDING_NAME_KEYWORDS = ("embed", "bge", "e5", "minilm")

OLLAMA_CHAT_STAT_KEYS = (
    "prompt_eval_count",
    "prompt_eval_duration",
    "eval_count",
    "eval_duration",
    "total_duration",
    "load_duration",
)


def format_bytes(size_bytes: int | None) -> str:
    """Format byte size into human-readable disk usage (e.g. '2.6 GB', '274 MB')."""
    if not size_bytes or size_bytes <= 0:
        return "Unknown size"
    gb = size_bytes / (1024**3)
    if gb >= 1.0:
        return f"{gb:.1f} GB"
    mb = size_bytes / (1024**2)
    if mb >= 1.0:
        return f"{mb:.0f} MB"
    kb = size_bytes / 1024
    return f"{kb:.0f} KB"


def infer_capabilities_from_name(model_name: str) -> list[str]:
    """Fallback heuristic when Ollama does not report capabilities (§4.4).

    Tags containing 'embed', 'bge', 'e5', or 'minilm' are treated as embedding models;
    all others default to completion models.
    """
    lowered = model_name.lower()
    if any(keyword in lowered for keyword in EMBEDDING_NAME_KEYWORDS):
        return ["embedding"]
    return ["completion"]


def normalize_model_tag(tag: str) -> str:
    """Canonicalize Ollama model tags so bare names match '<name>:latest'."""
    cleaned = tag.strip()
    if not cleaned:
        return ""
    if ":" not in cleaned:
        return f"{cleaned}:latest"
    return cleaned


def models_match(tag_a: str | None, tag_b: str | None) -> bool:
    """Return True when two model tags refer to the same canonical model tag."""
    if not tag_a or not tag_b:
        return False
    return normalize_model_tag(tag_a).lower() == normalize_model_tag(tag_b).lower()


def resolve_installed_model(
    requested_tag: str, installed_models: list[dict[str, Any]]
) -> dict[str, Any] | None:
    """Locate an installed model matching requested_tag (handling ':latest' aliases)."""
    cleaned = requested_tag.strip()
    if not cleaned:
        return None
    # Exact match first
    for model in installed_models:
        if str(model.get("name", "")).lower() == cleaned.lower():
            return model
    # Canonical tag match (e.g. 'nomic-embed-text' <-> 'nomic-embed-text:latest')
    for model in installed_models:
        if models_match(cleaned, str(model.get("name", ""))):
            return model
    return None


class LLMBackend(ABC):
    """Abstract base class for local LLM and embedding platforms (§4.4)."""

    @abstractmethod
    def list_models(self) -> list[dict[str, Any]]:
        """List installed models with capabilities, verification flag, and size."""

    @abstractmethod
    def chat(
        self,
        messages: list[dict[str, str]],
        model: str,
        options: dict[str, Any] | None = None,
        stats: dict[str, Any] | None = None,
    ) -> Iterator[str]:
        """Generate or stream chat completion tokens."""

    @abstractmethod
    def embed(self, texts: list[str], model: str) -> list[list[float]]:
        """Generate embedding vectors for a batch of input strings."""

    @abstractmethod
    def pull_model(self, model: str) -> Iterator[dict[str, Any]]:
        """Download a model with streaming progress events (user-initiated only)."""
