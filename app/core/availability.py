"""Model availability, 30-second cache, fallback resolution, and metadata checks (§4.7)."""

from __future__ import annotations

from dataclasses import dataclass, field
import threading
import time
from typing import Any

from app.config import AppConfig
from app.core.backends.base import LLMBackend, resolve_installed_model
from app.core.errors import (
    BackendUnavailableError,
    ChatModelMissingError,
    CollectionMetadataMismatchError,
    EmbeddingModelMissingError,
    NoModelsInstalledError,
)
from app.core.profiles import Profile
from app.core.vectorstore import VectorStore


def _names_with_capability(models: list[dict[str, Any]], cap: str) -> list[str]:
    return [str(m["name"]) for m in models if cap in (m.get("capabilities") or [])]


def _find_model(
    tag: str, models: list[dict[str, Any]], cap: str
) -> dict[str, Any] | None:
    match = resolve_installed_model(tag, models)
    if match is not None and cap in (match.get("capabilities") or []):
        return match
    return None


@dataclass
class ProfileAvailability:
    """Evaluated readiness and warning/fallback status for a specific bot profile."""

    ollama_reachable: bool = True
    no_models_installed: bool = False
    has_warning: bool = False
    chat_blocked: bool = False
    upload_blocked: bool = False
    block_code: str | None = None
    block_message: str | None = None

    chat_model_installed: bool = True
    effective_chat_model: str = ""
    fallback_active: bool = False
    fallback_banner: str | None = None

    embedding_model_installed: bool = True
    metadata_ok: bool = True
    mismatch_details: dict[str, Any] | None = None

    available_chat_models: list[str] = field(default_factory=list)
    available_embedding_models: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ollama_reachable": self.ollama_reachable,
            "no_models_installed": self.no_models_installed,
            "has_warning": self.has_warning,
            "chat_blocked": self.chat_blocked,
            "upload_blocked": self.upload_blocked,
            "block_code": self.block_code,
            "block_message": self.block_message,
            "chat_model_installed": self.chat_model_installed,
            "effective_chat_model": self.effective_chat_model,
            "fallback_active": self.fallback_active,
            "fallback_banner": self.fallback_banner,
            "embedding_model_installed": self.embedding_model_installed,
            "metadata_ok": self.metadata_ok,
            "mismatch_details": self.mismatch_details,
            "available_chat_models": self.available_chat_models,
            "available_embedding_models": self.available_embedding_models,
            "warnings": self.warnings,
        }


class ModelAvailabilityManager:
    """Checks Ollama reachability, installed model capabilities, and fallback rules (§4.7)."""

    def __init__(self, backend: LLMBackend, config: AppConfig) -> None:
        self.backend = backend
        self.config = config
        self._lock = threading.Lock()
        self._cached_models: list[dict[str, Any]] | None = None
        self._cached_at: float = 0.0

    @property
    def recommended_commands(self) -> list[str]:
        return [
            f"ollama pull {self.config.default_chat_model}",
            f"ollama pull {self.config.default_embedding_model}",
        ]

    def invalidate_cache(self) -> None:
        with self._lock:
            self._cached_models = None
            self._cached_at = 0.0

    def get_installed_models(self, *, force_refresh: bool = False) -> list[dict[str, Any]]:
        """Return installed models using a thread-safe TTL cache (default 30s)."""
        now = time.monotonic()
        with self._lock:
            if (
                not force_refresh
                and self._cached_models is not None
                and (now - self._cached_at) < self.config.model_cache_ttl_seconds
            ):
                return list(self._cached_models)

        models = self.backend.list_models()
        with self._lock:
            self._cached_models = list(models)
            self._cached_at = time.monotonic()
        return models

    def get_system_status(self, *, force_refresh: bool = False) -> dict[str, Any]:
        """Evaluate global Ollama health and installed models categorized by capability."""
        ollama_url = getattr(self.backend, "base_url", self.config.ollama_base_url)
        try:
            models = self.get_installed_models(force_refresh=force_refresh)
        except BackendUnavailableError as exc:
            self.invalidate_cache()
            return {
                "status": "degraded",
                "ollama_reachable": False,
                "ollama_url": ollama_url,
                "banner_message": exc.message,
                "details": exc.details,
                "no_models_installed": False,
                "models": [],
                "completion_models": [],
                "embedding_models": [],
                "default_chat_model": self.config.default_chat_model,
                "default_embedding_model": self.config.default_embedding_model,
                "allow_chat_fallback": self.config.allow_chat_fallback,
                "recommended_commands": self.recommended_commands,
            }

        completion_models = [
            m for m in models if "completion" in (m.get("capabilities") or [])
        ]
        embedding_models = [
            m for m in models if "embedding" in (m.get("capabilities") or [])
        ]
        no_models = len(models) == 0

        return {
            "status": "ok" if not no_models else "setup_required",
            "ollama_reachable": True,
            "ollama_url": ollama_url,
            "banner_message": None,
            "no_models_installed": no_models,
            "models": models,
            "completion_models": completion_models,
            "embedding_models": embedding_models,
            "default_chat_model": self.config.default_chat_model,
            "default_embedding_model": self.config.default_embedding_model,
            "allow_chat_fallback": self.config.allow_chat_fallback,
            "recommended_commands": self.recommended_commands,
        }

    def resolve_embedding_model(
        self,
        profile: Profile,
        *,
        _models: list[dict[str, Any]] | None = None,
    ) -> str:
        """Resolve profile.embedding_model to an installed embedding model tag or raise."""
        models = _models if _models is not None else self.get_installed_models()
        if len(models) == 0:
            raise NoModelsInstalledError(self.recommended_commands)

        embed_match = _find_model(profile.embedding_model, models, "embedding")
        if embed_match is None:
            raise EmbeddingModelMissingError(
                profile.embedding_model,
                available_embedding_models=_names_with_capability(models, "embedding"),
            )
        return str(embed_match["name"])

    def _resolve_chat_model(
        self, profile: Profile, models: list[dict[str, Any]]
    ) -> tuple[str | None, bool, bool, str | None]:
        """Resolve chat model and fallback policy.

        Returns:
            (effective_chat_model_or_none, chat_model_installed, fallback_active, fallback_banner)
        """
        chat_match = _find_model(profile.chat_model, models, "completion")
        if chat_match is not None:
            return str(chat_match["name"]), True, False, None

        if self.config.allow_chat_fallback:
            fallback_match = _find_model(
                self.config.default_chat_model, models, "completion"
            )
            if fallback_match is not None:
                banner = (
                    f"Using {self.config.default_chat_model} because "
                    f"{profile.chat_model} is not installed."
                )
                return str(fallback_match["name"]), False, True, banner

        return None, False, False, None

    def evaluate_profile(
        self,
        profile: Profile,
        vectorstore: VectorStore,
        *,
        force_refresh: bool = False,
    ) -> ProfileAvailability:
        """Evaluate all §4.7 rules for a given profile."""
        try:
            models = self.get_installed_models(force_refresh=force_refresh)
        except BackendUnavailableError as exc:
            self.invalidate_cache()
            return ProfileAvailability(
                ollama_reachable=False,
                has_warning=True,
                chat_blocked=True,
                upload_blocked=True,
                block_code="ollama_unreachable",
                block_message=exc.message,
                effective_chat_model=profile.chat_model,
                warnings=[exc.message],
            )

        if len(models) == 0:
            return ProfileAvailability(
                ollama_reachable=True,
                no_models_installed=True,
                has_warning=True,
                chat_blocked=True,
                upload_blocked=True,
                block_code="no_models_installed",
                block_message=(
                    "No Ollama models are installed. Pull the recommended models to begin."
                ),
                effective_chat_model=profile.chat_model,
                warnings=["No Ollama models installed."],
            )

        result = ProfileAvailability(
            ollama_reachable=True,
            effective_chat_model=profile.chat_model,
            available_chat_models=_names_with_capability(models, "completion"),
            available_embedding_models=_names_with_capability(models, "embedding"),
        )

        # 1. Check embedding model installation (NEVER falls back — §4.7 row 4)
        if _find_model(profile.embedding_model, models, "embedding") is None:
            result.embedding_model_installed = False
            result.has_warning = True
            result.chat_blocked = True
            result.upload_blocked = True
            result.block_code = "embedding_model_missing"
            msg = (
                f"Embedding model '{profile.embedding_model}' is not installed. "
                "Chat and uploads are blocked until you download it or switch to an installed embedding model."
            )
            result.block_message = msg
            result.warnings.append(msg)

        # 2. Check collection metadata compatibility (§4.7 row 5)
        try:
            vectorstore.validate_compatibility(profile.id, profile.embedding_model)
        except CollectionMetadataMismatchError as exc:
            result.metadata_ok = False
            result.has_warning = True
            result.chat_blocked = True
            result.upload_blocked = True
            if result.block_code is None:
                result.block_code = exc.code
                result.block_message = exc.message
            result.mismatch_details = exc.extra
            result.warnings.append(exc.message)

        # 3. Check chat model installation & explicit fallback (§4.7 row 3)
        effective_chat, chat_installed, fallback_active, fallback_banner = (
            self._resolve_chat_model(profile, models)
        )
        if chat_installed and effective_chat is not None:
            result.chat_model_installed = True
            result.effective_chat_model = effective_chat
        else:
            result.chat_model_installed = False
            result.has_warning = True
            if effective_chat is not None and fallback_banner is not None:
                result.effective_chat_model = effective_chat
                result.fallback_active = fallback_active
                result.fallback_banner = fallback_banner
                result.warnings.append(fallback_banner)
            else:
                result.chat_blocked = True
                msg = f"Chat model '{profile.chat_model}' is not installed."
                if result.block_code is None:
                    result.block_code = "chat_model_missing"
                    result.block_message = msg
                result.warnings.append(msg)

        return result

    def require_ready_for_ingestion(
        self, profile: Profile, vectorstore: VectorStore
    ) -> str:
        """Enforce §4.7 availability checks before document ingestion; returns canonical embed tag."""
        effective_embed = self.resolve_embedding_model(profile)
        vectorstore.validate_compatibility(profile.id, profile.embedding_model)
        return effective_embed

    def require_ready_for_chat(
        self, profile: Profile, vectorstore: VectorStore
    ) -> tuple[str, str, bool, str | None]:
        """Enforce §4.7 availability checks before chat.

        Returns:
            (effective_chat_model, effective_embedding_model, fallback_active, fallback_banner)
        """
        models = self.get_installed_models()
        effective_embed = self.resolve_embedding_model(profile, _models=models)
        vectorstore.validate_compatibility(profile.id, profile.embedding_model)

        effective_chat, _, fallback_active, fallback_banner = self._resolve_chat_model(
            profile, models
        )
        if effective_chat is not None:
            return effective_chat, effective_embed, fallback_active, fallback_banner

        raise ChatModelMissingError(
            profile.chat_model,
            available_chat_models=_names_with_capability(models, "completion"),
        )
