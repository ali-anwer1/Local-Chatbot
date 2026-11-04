"""Shared pytest fixtures and deterministic mock LLM/embedding backend."""

from __future__ import annotations

from collections.abc import Iterator
import math
from pathlib import Path
import re
from typing import Any
import pytest

from app.config import AppConfig
from app.core.backends.base import LLMBackend, format_bytes
from app.core.errors import (
    BackendUnavailableError,
    ModelDownloadOfflineError,
    ModelLoadError,
)
from app.core.service import ChatbotService


VOCAB_TOPICS = [
    ("cortex", "a78ae", "processor", "ram", "lpddr5", "emmc", "storage", "8", "64"),
    ("voltage", "dc", "power", "12", "48", "34", "watt", "electrical", "temperature"),
    ("poe", "eth1", "eth0", "galvanic", "isolation", "serial", "rs-485", "modbus", "can-fd"),
    ("rtc", "battery", "cr2032w", "preventive", "schedule", "heatsink", "dust"),
    ("thermal", "pad", "ast-tpad-15", "torque", "0.65", "m3", "silicone"),
    ("firmware", "recovery", "recov", "192.168.200.1", "8443", "7", "seconds"),
    ("warranty", "36-month", "36", "astracare", "60-month", "60", "tamper"),
    ("sla", "p1", "rma", "hours", "14", "calendar", "replacement"),
    ("led", "stat", "amber", "red", "tpm", "e-409", "attestation"),
    ("clearance", "35", "mm", "airflow", "din-rail", "vertical"),
    ("alpha", "secret", "project-alpha", "cobalt"),
    ("beta", "confidential", "project-beta", "crimson"),
]


class DeterministicMockBackend(LLMBackend):
    """Controllable in-memory LLMBackend for fast, deterministic offline unit tests."""

    def __init__(
        self,
        *,
        models: list[dict[str, Any]] | None = None,
        reachable: bool = True,
    ) -> None:
        self.reachable = reachable
        self.base_url = "http://127.0.0.1:11434"
        self.models: list[dict[str, Any]] = (
            list(models)
            if models is not None
            else [
                {
                    "name": "qwen3.5:4b",
                    "size_bytes": int(2.6 * 1024**3),
                    "size_human": format_bytes(int(2.6 * 1024**3)),
                    "parameter_size": "4.0B",
                    "quantization": "Q4_K_M",
                    "capabilities": ["completion"],
                    "capabilities_verified": True,
                },
                {
                    "name": "llama3.2:3b",
                    "size_bytes": int(2.0 * 1024**3),
                    "size_human": format_bytes(int(2.0 * 1024**3)),
                    "parameter_size": "3.2B",
                    "quantization": "Q4_K_M",
                    "capabilities": ["completion"],
                    "capabilities_verified": True,
                },
                {
                    "name": "nomic-embed-text:latest",
                    "size_bytes": 274 * 1024**2,
                    "size_human": format_bytes(274 * 1024**2),
                    "parameter_size": "137M",
                    "quantization": "F16",
                    "capabilities": ["embedding"],
                    "capabilities_verified": True,
                },
                {
                    "name": "mxbai-embed-large:latest",
                    "size_bytes": 670 * 1024**2,
                    "size_human": format_bytes(670 * 1024**2),
                    "parameter_size": "334M",
                    "quantization": "F16",
                    "capabilities": ["embedding"],
                    "capabilities_verified": True,
                },
            ]
        )
        self.embed_dimensions: dict[str, int] = {
            "nomic-embed-text": len(VOCAB_TOPICS),
            "nomic-embed-text:latest": len(VOCAB_TOPICS),
            "mxbai-embed-large": len(VOCAB_TOPICS) + 4,
            "mxbai-embed-large:latest": len(VOCAB_TOPICS) + 4,
        }
        self.fail_load_models: set[str] = set()
        self.fail_mid_stream_models: set[str] = set()
        self.pull_offline: bool = False
        self.next_chat_reply: str | None = None
        self.next_chat_stats: dict[str, Any] | None = None

        self.chat_calls: list[dict[str, Any]] = []
        self.embed_calls: list[dict[str, Any]] = []

    def list_models(self) -> list[dict[str, Any]]:
        if not self.reachable:
            raise BackendUnavailableError(self.base_url, details="Connection refused")
        return list(self.models)

    def embed(self, texts: list[str], model: str) -> list[list[float]]:
        if not self.reachable:
            raise BackendUnavailableError(self.base_url, details="Connection refused")
        if model in self.fail_load_models:
            raise ModelLoadError(model, details="CUDA/Vulkan error: failed to allocate 2.4 GiB buffer")

        self.embed_calls.append({"model": model, "count": len(texts)})
        dim = self.embed_dimensions.get(model, len(VOCAB_TOPICS))

        vectors: list[list[float]] = []
        for text in texts:
            tokens = set(re.findall(r"[a-z0-9_.-]+", text.lower()))
            vec = [0.0] * dim
            for idx, topic_words in enumerate(VOCAB_TOPICS):
                if idx >= dim:
                    break
                hits = sum(1 for w in topic_words if w in tokens)
                vec[idx] = float(hits)

            norm = math.sqrt(sum(v * v for v in vec))
            if norm > 0:
                vec = [v / norm for v in vec]
            else:
                # Orthogonal fallback axis if extra dims exist, otherwise uniform tiny vector
                # Wait: for out-of-scope text with 0 keyword hits, put energy in a dedicated
                # "unrelated" dimension or yield zero overlap with indexed topic vectors!
                vec = [0.0] * dim
            vectors.append(vec)

        # Ensure zero-norm vectors have a dedicated orthogonal component so cosine distance = 1.0 (score = 0.0)
        # by appending one extra orthogonal slot to every vector consistently.
        padded: list[list[float]] = []
        for vec in vectors:
            has_signal = any(v > 0 for v in vec)
            padded.append(vec + ([0.0] if has_signal else [1.0]))
        return padded

    def chat(
        self,
        messages: list[dict[str, str]],
        model: str,
        options: dict[str, Any] | None = None,
        stats: dict[str, Any] | None = None,
    ) -> Iterator[str]:
        if not self.reachable:
            raise BackendUnavailableError(self.base_url, details="Connection refused")
        if model in self.fail_load_models:
            raise ModelLoadError(model, details="llama_model_load: error loading model: not enough RAM")

        if stats is not None and self.next_chat_stats is not None:
            stats.update(self.next_chat_stats)
        self.next_chat_stats = None

        self.chat_calls.append(
            {"model": model, "messages": messages, "options": options}
        )

        if model in self.fail_mid_stream_models:
            yield "Partial "
            raise ModelLoadError(model, details="GGML_ASSERT: out of memory during KV cache allocation")

        if self.next_chat_reply is not None:
            custom_reply = self.next_chat_reply
            self.next_chat_reply = None
            yield custom_reply
            return

        user_prompt = messages[-1]["content"]
        if "[1]" not in user_prompt:
            yield f"Hello! I can help you explore this profile's uploaded documents ({model})."
            return

        # Echo concise grounded answer citing [1] with key snippet from passage [1]
        after_one = user_prompt.split("[1]", 1)[1]
        first_passage = after_one.split("\n\nQuestion:", 1)[0].strip()[:140]
        reply_tokens = [
            f"Based on the documentation ({model}): ",
            first_passage,
            " [1].",
        ]
        yield from reply_tokens

    def pull_model(self, model: str) -> Iterator[dict[str, Any]]:
        if not self.reachable:
            raise BackendUnavailableError(self.base_url, details="Connection refused")
        if self.pull_offline:
            raise ModelDownloadOfflineError(
                details='pull model manifest: Get "https://registry.ollama.ai/v2/...": dial tcp: lookup registry.ollama.ai: no such host'
            )
        yield {"status": "pulling manifest", "digest": None, "total": 100, "completed": 10, "percent": 10.0}
        yield {"status": "downloading", "digest": "sha256:abc", "total": 100, "completed": 100, "percent": 100.0}
        self.models.append(
            {
                "name": model,
                "size_bytes": 500 * 1024**2,
                "size_human": "500 MB",
                "parameter_size": "1B",
                "quantization": "Q4_K_M",
                "capabilities": ["embedding"] if "embed" in model else ["completion"],
                "capabilities_verified": True,
            }
        )


@pytest.fixture()
def test_config(tmp_path: Path) -> AppConfig:
    return AppConfig(
        host="127.0.0.1",
        port=8000,
        data_dir=tmp_path / "data",
        ollama_base_url="http://127.0.0.1:11434",
        default_chat_model="qwen3.5:4b",
        default_embedding_model="nomic-embed-text",
        allow_chat_fallback=True,
        model_cache_ttl_seconds=30.0,
    )


@pytest.fixture()
def mock_backend() -> DeterministicMockBackend:
    return DeterministicMockBackend()


@pytest.fixture()
def chatbot_service(
    test_config: AppConfig, mock_backend: DeterministicMockBackend
) -> ChatbotService:
    return ChatbotService(config=test_config, backend=mock_backend)
