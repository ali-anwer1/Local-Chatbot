"""Configuration loader for Local RAG Chatbot."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"
ALLOWED_HOSTS = {"127.0.0.1", "localhost"}


@dataclass(frozen=True)
class AppConfig:
    """Immutable application configuration loaded from config.yaml."""

    host: str = "127.0.0.1"
    port: int = 8000
    data_dir: Path = PROJECT_ROOT / "data"
    ollama_base_url: str = "http://127.0.0.1:11434"
    default_chat_model: str = "qwen3.5:4b"
    default_embedding_model: str = "nomic-embed-text"
    allow_chat_fallback: bool = True
    model_cache_ttl_seconds: float = 30.0

    def __post_init__(self) -> None:
        if self.host not in ALLOWED_HOSTS:
            raise ValueError(
                f"Invalid host '{self.host}'. Server must bind strictly to 127.0.0.1."
            )


def load_config(config_path: Path | str | None = None) -> AppConfig:
    """Load configuration from YAML file, falling back to safe local defaults."""
    path = Path(config_path) if config_path is not None else DEFAULT_CONFIG_PATH
    raw: dict[str, Any] = {}
    if path.exists():
        with path.open("r", encoding="utf-8") as fh:
            loaded = yaml.safe_load(fh)
            if isinstance(loaded, dict):
                raw = loaded

    host = str(raw.get("host", "127.0.0.1")).strip()
    if host == "localhost":
        host = "127.0.0.1"

    data_dir_raw = raw.get("data_dir", "data")
    data_dir = Path(data_dir_raw)
    if not data_dir.is_absolute():
        base_dir = path.resolve().parent if path.exists() else PROJECT_ROOT
        data_dir = (base_dir / data_dir).resolve()

    ollama_url = str(raw.get("ollama_base_url", "http://127.0.0.1:11434")).rstrip("/")

    return AppConfig(
        host=host,
        port=int(raw.get("port", 8000)),
        data_dir=data_dir,
        ollama_base_url=ollama_url,
        default_chat_model=str(raw.get("default_chat_model", "qwen3.5:4b")).strip(),
        default_embedding_model=str(
            raw.get("default_embedding_model", "nomic-embed-text")
        ).strip(),
        allow_chat_fallback=bool(raw.get("allow_chat_fallback", True)),
        model_cache_ttl_seconds=float(raw.get("model_cache_ttl_seconds", 30.0)),
    )
