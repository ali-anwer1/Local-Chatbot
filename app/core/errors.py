"""Structured domain exceptions for Local RAG Chatbot.

All exceptions carry user-safe messages and structured metadata so the API and UI
never expose raw Python stack traces or cryptic backend failures (§4.7, §7.9).
"""

from __future__ import annotations

from typing import Any


class ChatbotError(Exception):
    """Base domain error with structured code, user message, and optional details."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        details: str | None = None,
        status_hint: int = 400,
        extra: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details
        self.status_hint = status_hint
        self.extra: dict[str, Any] = extra or {}

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "code": self.code,
            "message": self.message,
        }
        if self.details:
            payload["details"] = self.details
        if self.extra:
            payload.update(self.extra)
        return payload


class BackendUnavailableError(ChatbotError):
    """Raised when Ollama cannot be reached at the configured URL."""

    def __init__(self, url: str, details: str | None = None) -> None:
        super().__init__(
            code="ollama_unreachable",
            message=f"Ollama is not reachable at {url}. Start Ollama and click Retry.",
            details=details,
            status_hint=503,
            extra={"ollama_url": url},
        )


class NoModelsInstalledError(ChatbotError):
    """Raised when Ollama is running but zero models are installed."""

    def __init__(self, recommended_commands: list[str] | None = None) -> None:
        cmds = recommended_commands or []
        super().__init__(
            code="no_models_installed",
            message="No Ollama models are installed yet. Pull the recommended models to get started.",
            status_hint=409,
            extra={"recommended_commands": cmds},
        )


class ChatModelMissingError(ChatbotError):
    """Raised when a profile's chat model is missing and fallback cannot be used."""

    def __init__(
        self,
        missing_model: str,
        available_chat_models: list[str] | None = None,
        details: str | None = None,
    ) -> None:
        super().__init__(
            code="chat_model_missing",
            message=f"Chat model '{missing_model}' is not installed.",
            details=details,
            status_hint=409,
            extra={
                "missing_model": missing_model,
                "available_chat_models": available_chat_models or [],
            },
        )


class EmbeddingModelMissingError(ChatbotError):
    """Raised when a profile's embedding model is missing (never falls back)."""

    def __init__(
        self,
        missing_model: str,
        available_embedding_models: list[str] | None = None,
        details: str | None = None,
    ) -> None:
        super().__init__(
            code="embedding_model_missing",
            message=(
                f"Embedding model '{missing_model}' is not installed. "
                "Download the missing model or switch to an installed embedding model (requires re-indexing)."
            ),
            details=details,
            status_hint=409,
            extra={
                "missing_model": missing_model,
                "available_embedding_models": available_embedding_models or [],
            },
        )


class CollectionMetadataMismatchError(ChatbotError):
    """Raised when the vector collection's stored model or dimension mismatches the profile."""

    def __init__(
        self,
        *,
        expected_model: str,
        stored_model: str | None,
        expected_dimension: int | None = None,
        stored_dimension: int | None = None,
        estimated_chunks: int = 0,
    ) -> None:
        reasons: list[str] = []
        if stored_model != expected_model:
            reasons.append(
                f"profile expects embedding model '{expected_model}' but index was built with '{stored_model or 'unknown'}'"
            )
        if (
            expected_dimension is not None
            and stored_dimension is not None
            and expected_dimension != stored_dimension
        ):
            reasons.append(
                f"embedding dimension is {expected_dimension} but index stores {stored_dimension}-dimensional vectors"
            )
        reason_str = "; ".join(reasons) if reasons else "embedding metadata mismatch"
        super().__init__(
            code="collection_metadata_mismatch",
            message=(
                f"Knowledge base index metadata does not match ({reason_str}). "
                "Re-index the knowledge base to prevent corrupted retrieval."
            ),
            status_hint=409,
            extra={
                "expected_model": expected_model,
                "stored_model": stored_model,
                "expected_dimension": expected_dimension,
                "stored_dimension": stored_dimension,
                "estimated_chunks": estimated_chunks,
            },
        )


class ModelLoadError(ChatbotError):
    """Raised when an installed model fails to load or run (e.g. out of memory)."""

    def __init__(self, model: str, details: str | None = None) -> None:
        super().__init__(
            code="model_load_failed",
            message=f"{model} failed to load, likely insufficient memory. Try a smaller model.",
            details=details,
            status_hint=500,
            extra={"model": model},
        )


class ModelDownloadOfflineError(ChatbotError):
    """Raised when a user-initiated model download fails due to lack of internet."""

    def __init__(self, details: str | None = None) -> None:
        super().__init__(
            code="download_offline",
            message="No internet connection; download unavailable offline",
            details=details,
            status_hint=503,
        )


class ReindexConfirmationRequiredError(ChatbotError):
    """Raised when changing embedding_model without explicit confirmation."""

    def __init__(
        self,
        *,
        current_embedding_model: str,
        requested_embedding_model: str,
        estimated_chunks: int,
        document_count: int,
    ) -> None:
        super().__init__(
            code="reindex_confirmation_required",
            message=(
                f"Changing the embedding model from '{current_embedding_model}' to "
                f"'{requested_embedding_model}' will re-index the knowledge base "
                f"(~{estimated_chunks} chunks across {document_count} document(s)). "
                "Confirm to proceed."
            ),
            status_hint=409,
            extra={
                "current_embedding_model": current_embedding_model,
                "requested_embedding_model": requested_embedding_model,
                "estimated_chunks": estimated_chunks,
                "document_count": document_count,
            },
        )


class ProfileNotFoundError(ChatbotError):
    """Raised when a requested bot profile does not exist."""

    def __init__(self, profile_id: str) -> None:
        super().__init__(
            code="profile_not_found",
            message=f"Profile '{profile_id}' was not found.",
            status_hint=404,
            extra={"profile_id": profile_id},
        )


class DocumentNotFoundError(ChatbotError):
    """Raised when a requested document does not exist in a profile."""

    def __init__(self, doc_id: str, profile_id: str) -> None:
        super().__init__(
            code="document_not_found",
            message=f"Document '{doc_id}' was not found in profile '{profile_id}'.",
            status_hint=404,
            extra={"doc_id": doc_id, "profile_id": profile_id},
        )


class ChatNotFoundError(ChatbotError):
    """Raised when a requested saved chat session does not exist in a profile."""

    def __init__(self, chat_id: str, profile_id: str) -> None:
        super().__init__(
            code="chat_not_found",
            message=f"Saved chat '{chat_id}' was not found in profile '{profile_id}'.",
            status_hint=404,
            extra={"chat_id": chat_id, "profile_id": profile_id},
        )


class ValidationError(ChatbotError):
    """Raised when user input fails domain validation."""

    def __init__(self, message: str, details: str | None = None) -> None:
        super().__init__(
            code="validation_error",
            message=message,
            details=details,
            status_hint=422,
        )
