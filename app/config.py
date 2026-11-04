"""Configuration loader and mutable runtime app-settings store for Local RAG Chatbot."""

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
import os
from pathlib import Path
import tempfile
import threading
from typing import Any
import yaml

from app.core.errors import ValidationError


logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"
ALLOWED_HOSTS = {"127.0.0.1", "localhost"}
VALID_EMBEDDING_DEVICES = ("cpu", "auto")


def atomic_write_json(path: Path, data: Any) -> None:
    """Write JSON atomically via a temp file in the same directory and os.replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as fh:
            tmp_path = Path(fh.name)
            json.dump(data, fh, indent=2, ensure_ascii=False)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_path, path)
    except BaseException:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)
        raise


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
    embedding_device: str = "cpu"
    keep_alive: str = "30m"

    def __post_init__(self) -> None:
        if self.host not in ALLOWED_HOSTS:
            raise ValueError(
                f"Invalid host '{self.host}'. Server must bind strictly to 127.0.0.1."
            )
        if self.embedding_device not in VALID_EMBEDDING_DEVICES:
            raise ValueError(
                f"Invalid embedding_device '{self.embedding_device}'. Must be 'cpu' or 'auto'."
            )
        if not isinstance(self.keep_alive, str) or not self.keep_alive.strip():
            raise ValueError("keep_alive must be a non-empty duration string (e.g. '30m').")


def _validate_embedding_device(val: Any) -> str:
    if not isinstance(val, str) or val not in VALID_EMBEDDING_DEVICES:
        raise ValidationError(
            f"Invalid embedding_device '{val}'. Must be 'cpu' or 'auto'."
        )
    return val


class AppSettings:
    """Mutable machine-wide runtime settings persisted to data/app_settings.json."""

    def __init__(
        self,
        *,
        embedding_device: str = "cpu",
        keep_alive: str = "30m",
        persist_path: Path | None = None,
    ) -> None:
        self._lock = threading.Lock()
        self._embedding_device: str = _validate_embedding_device(embedding_device)
        self._keep_alive: str = keep_alive
        self._persist_path: Path | None = persist_path

    @property
    def embedding_device(self) -> str:
        with self._lock:
            return self._embedding_device

    @property
    def keep_alive(self) -> str:
        with self._lock:
            return self._keep_alive

    def to_dict(self) -> dict[str, Any]:
        with self._lock:
            return {
                "embedding_device": self._embedding_device,
                "keep_alive": self._keep_alive,
            }

    def update(self, payload: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(payload, dict) or "embedding_device" not in payload:
            raise ValidationError("embedding_device is required and must be 'cpu' or 'auto'.")

        raw_device = _validate_embedding_device(payload["embedding_device"])

        with self._lock:
            self._embedding_device = raw_device
            if self._persist_path is not None:
                atomic_write_json(
                    self._persist_path,
                    {"embedding_device": self._embedding_device},
                )
            return {
                "embedding_device": self._embedding_device,
                "keep_alive": self._keep_alive,
            }

    @classmethod
    def from_config(cls, config: AppConfig) -> AppSettings:
        """Initialize from AppConfig defaults, applying overrides from data/app_settings.json."""
        persist_path = Path(config.data_dir) / "app_settings.json"
        embedding_device = config.embedding_device
        if persist_path.exists():
            try:
                raw = json.loads(persist_path.read_text(encoding="utf-8"))
                if isinstance(raw, dict) and "embedding_device" in raw:
                    try:
                        embedding_device = _validate_embedding_device(raw["embedding_device"])
                    except ValidationError:
                        logger.warning(
                            "Ignoring invalid embedding_device %r in %s",
                            raw["embedding_device"],
                            persist_path,
                        )
            except Exception as exc:
                logger.warning(
                    "Skipping unreadable app settings at %s: %s",
                    persist_path,
                    exc,
                )
        return cls(
            embedding_device=embedding_device,
            keep_alive=config.keep_alive,
            persist_path=persist_path,
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
    embedding_device = str(raw.get("embedding_device", "cpu")).strip()
    keep_alive = str(raw.get("keep_alive", "30m")).strip() or "30m"

    defaults = AppConfig()
    return AppConfig(
        host=host,
        port=int(raw.get("port", defaults.port)),
        data_dir=data_dir,
        ollama_base_url=ollama_url,
        default_chat_model=str(
            raw.get("default_chat_model", defaults.default_chat_model)
        ).strip(),
        default_embedding_model=str(
            raw.get("default_embedding_model", defaults.default_embedding_model)
        ).strip(),
        allow_chat_fallback=bool(raw.get("allow_chat_fallback", defaults.allow_chat_fallback)),
        model_cache_ttl_seconds=float(raw.get("model_cache_ttl_seconds", defaults.model_cache_ttl_seconds)),
        embedding_device=embedding_device,
        keep_alive=keep_alive,
    )
