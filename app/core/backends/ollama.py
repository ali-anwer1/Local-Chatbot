"""Ollama local REST API backend implementation (§4.4, §4.7)."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
import json
import threading
from typing import Any
import httpx

from app.config import VALID_EMBEDDING_DEVICES, AppSettings
from app.core.backends.base import (
    LLMBackend,
    format_bytes,
    infer_capabilities_from_name,
)
from app.core.errors import (
    BackendUnavailableError,
    ChatModelMissingError,
    ChatbotError,
    EmbeddingModelMissingError,
    ModelDownloadOfflineError,
    ModelLoadError,
)


PASSTHROUGH_RUNTIME_OPTIONS = {
    "num_ctx",
    "num_thread",
    "num_gpu",
    "temperature",
    "top_p",
    "top_k",
    "seed",
    "num_predict",
    "repeat_penalty",
}

NETWORK_ERROR_INDICATORS = (
    "dial tcp",
    "no such host",
    "lookup ",
    "network is unreachable",
    "no route to host",
    "tls handshake timeout",
    "i/o timeout",
    "connection reset",
    "temporary failure in name resolution",
    "getaddrinfo",
    "wsano_data",
    "wsahost_not_found",
    "offline",
)

EMBED_BATCH_SIZE = 32
OLLAMA_CHAT_STAT_KEYS = (
    "prompt_eval_count",
    "prompt_eval_duration",
    "eval_count",
    "eval_duration",
)


class OllamaBackend(LLMBackend):
    """Communicates with a local Ollama server over HTTP."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:11434",
        timeout_seconds: float = 120.0,
        connect_timeout_seconds: float = 5.0,
        transport: httpx.BaseTransport | None = None,
        *,
        embedding_device: str = "cpu",
        keep_alive: str = "30m",
        app_settings: AppSettings | None = None,
    ) -> None:
        if embedding_device not in VALID_EMBEDDING_DEVICES:
            raise ValueError(
                f"Invalid embedding_device '{embedding_device}'. Must be 'cpu' or 'auto'."
            )
        self.base_url = base_url.rstrip("/")
        self.timeout = httpx.Timeout(timeout_seconds, connect=connect_timeout_seconds)
        self._transport = transport
        self._embedding_device = embedding_device
        self._keep_alive = keep_alive
        self._app_settings = app_settings
        self._thread_local = threading.local()
        self._last_chat_stats: dict[str, Any] | None = None
        # Cache /api/show metadata keyed by (model_name, digest) to avoid redundant calls
        self._show_cache: dict[tuple[str, str], dict[str, Any]] = {}
        # Cache raw capabilities per model name so chat() knows if 'thinking' is present
        self._raw_capabilities: dict[str, set[str]] = {}

    def bind_app_settings(self, app_settings: AppSettings) -> None:
        """Bind a shared mutable AppSettings holder so runtime setting changes take effect immediately."""
        self._app_settings = app_settings

    @property
    def embedding_device(self) -> str:
        if self._app_settings is not None:
            return self._app_settings.embedding_device
        return self._embedding_device

    @embedding_device.setter
    def embedding_device(self, value: str) -> None:
        if value not in VALID_EMBEDDING_DEVICES:
            raise ValueError(
                f"Invalid embedding_device '{value}'. Must be 'cpu' or 'auto'."
            )
        self._embedding_device = value
        if self._app_settings is not None:
            self._app_settings.update({"embedding_device": value})

    @property
    def keep_alive(self) -> str:
        if self._app_settings is not None:
            return self._app_settings.keep_alive
        return self._keep_alive

    @keep_alive.setter
    def keep_alive(self, value: str) -> None:
        self._keep_alive = value

    def _set_last_chat_stats(self, stats: dict[str, Any] | None) -> None:
        self._thread_local.last_chat_stats = stats
        self._last_chat_stats = stats

    def _record_chat_stats(self, chunk: dict[str, Any]) -> None:
        stats: dict[str, int] = {}
        for key in OLLAMA_CHAT_STAT_KEYS:
            val = chunk.get(key)
            if isinstance(val, (int, float)) and not isinstance(val, bool):
                stats[key] = int(val)
        self._set_last_chat_stats(stats if stats else None)

    def get_last_chat_stats(self) -> dict[str, Any] | None:
        stats = getattr(self._thread_local, "last_chat_stats", self._last_chat_stats)
        return dict(stats) if isinstance(stats, dict) else None

    def _client(self, timeout: httpx.Timeout | None = None) -> httpx.Client:
        return httpx.Client(
            base_url=self.base_url,
            timeout=timeout or self.timeout,
            transport=self._transport,
        )

    @contextmanager
    def _translate_http_errors(
        self,
        *,
        model: str | None = None,
        offline_on_network_error: bool = False,
    ) -> Iterator[None]:
        try:
            yield
        except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
            raise BackendUnavailableError(self.base_url, details=str(exc)) from exc
        except httpx.NetworkError as exc:
            if offline_on_network_error:
                raise ModelDownloadOfflineError(details=str(exc)) from exc
            raise BackendUnavailableError(self.base_url, details=str(exc)) from exc
        except httpx.TimeoutException as exc:
            if model is not None:
                raise ModelLoadError(
                    model, details=f"Timed out waiting for model '{model}': {exc}"
                ) from exc
            raise BackendUnavailableError(self.base_url, details=str(exc)) from exc

    def _extract_error_message(self, response: httpx.Response) -> str:
        try:
            data = response.json()
            if isinstance(data, dict) and "error" in data:
                return str(data["error"])
        except Exception:
            pass
        return response.text.strip() or f"HTTP {response.status_code}"

    def _fetch_show_data(self, client: httpx.Client, model_name: str, digest: str = "") -> dict[str, Any]:
        cache_key = (model_name, digest)
        if digest and cache_key in self._show_cache:
            return self._show_cache[cache_key]

        resp = client.post("/api/show", json={"model": model_name})
        if resp.status_code == 404:
            err = self._extract_error_message(resp)
            raise ChatbotError(
                code="model_not_found",
                message=f"Model '{model_name}' is not installed.",
                details=err,
                status_hint=404,
            )
        if resp.status_code >= 400:
            return {}

        data = resp.json() if resp.content else {}
        if not isinstance(data, dict):
            data = {}
        if digest:
            self._show_cache[cache_key] = data
        return data

    def _parse_capabilities(
        self, model_name: str, show_data: dict[str, Any]
    ) -> tuple[list[str], bool, set[str]]:
        raw_caps = show_data.get("capabilities")
        if isinstance(raw_caps, list) and len(raw_caps) > 0:
            raw_set = {str(c).lower() for c in raw_caps}
            normalized: list[str] = []
            if "completion" in raw_set:
                normalized.append("completion")
            if "embedding" in raw_set:
                normalized.append("embedding")
            if normalized:
                self._raw_capabilities[model_name.lower()] = raw_set
                return normalized, True, raw_set

        # Fallback heuristic when Ollama version omits capabilities (§4.4)
        inferred = infer_capabilities_from_name(model_name)
        raw_set = set(inferred)
        self._raw_capabilities[model_name.lower()] = raw_set
        return inferred, False, raw_set

    def list_models(self) -> list[dict[str, Any]]:
        """List installed Ollama models with capabilities and size on disk."""
        with self._translate_http_errors(), self._client() as client:
            resp = client.get("/api/tags")
            if resp.status_code >= 400:
                err = self._extract_error_message(resp)
                raise BackendUnavailableError(self.base_url, details=err)
            payload = resp.json()
            raw_models = payload.get("models", []) if isinstance(payload, dict) else []

            results: list[dict[str, Any]] = []
            for entry in raw_models:
                if not isinstance(entry, dict):
                    continue
                name = str(entry.get("name") or entry.get("model") or "").strip()
                if not name:
                    continue
                size_bytes = int(entry.get("size") or 0)
                digest = str(entry.get("digest") or "")
                details = entry.get("details") if isinstance(entry.get("details"), dict) else {}

                show_data = self._fetch_show_data(client, name, digest=digest)
                show_details = (
                    show_data.get("details")
                    if isinstance(show_data.get("details"), dict)
                    else {}
                )
                capabilities, verified, raw_caps = self._parse_capabilities(name, show_data)
                if ":" in name and name.endswith(":latest"):
                    self._raw_capabilities[name[:-7].lower()] = raw_caps

                param_size = (
                    details.get("parameter_size")
                    or show_details.get("parameter_size")
                    or None
                )
                quantization = (
                    details.get("quantization_level")
                    or show_details.get("quantization_level")
                    or None
                )

                results.append(
                    {
                        "name": name,
                        "size_bytes": size_bytes,
                        "size_human": format_bytes(size_bytes),
                        "parameter_size": param_size,
                        "quantization": quantization,
                        "capabilities": capabilities,
                        "capabilities_verified": verified,
                    }
                )
            return results

    def model_info(self, model: str) -> dict[str, Any]:
        """Return size on disk, params, quantization, and capabilities for a model."""
        installed = self.list_models()
        for item in installed:
            if item["name"].lower() == model.strip().lower() or (
                ":" not in model and item["name"].lower() == f"{model.strip().lower()}:latest"
            ):
                return item

        # Fallback directly to /api/show if tag wasn't matched directly
        with self._translate_http_errors(), self._client() as client:
            show_data = self._fetch_show_data(client, model)
            capabilities, verified, _ = self._parse_capabilities(model, show_data)
            details = (
                show_data.get("details")
                if isinstance(show_data.get("details"), dict)
                else {}
            )
            size_bytes = int(show_data.get("size") or 0)
            return {
                "name": model,
                "size_bytes": size_bytes,
                "size_human": format_bytes(size_bytes),
                "parameter_size": details.get("parameter_size"),
                "quantization": details.get("quantization_level"),
                "capabilities": capabilities,
                "capabilities_verified": verified,
            }

    def _build_runtime_options(self, options: dict[str, Any] | None) -> dict[str, Any]:
        if not options:
            return {}
        return {
            key: val
            for key, val in options.items()
            if key in PASSTHROUGH_RUNTIME_OPTIONS and val is not None
        }

    def chat(
        self,
        messages: list[dict[str, str]],
        model: str,
        options: dict[str, Any] | None = None,
        stream: bool = True,
    ) -> Iterator[str]:
        """Call Ollama /api/chat and yield response text tokens."""
        self._set_last_chat_stats(None)
        runtime_options = self._build_runtime_options(options)
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": stream,
            "keep_alive": self.keep_alive,
        }
        if runtime_options:
            payload["options"] = runtime_options

        # Disable extended chain-of-thought delay by default (think=False) across all
        # thinking-capable models (e.g. qwen3.5:4b, qwen3, deepseek-r1) unless the user
        # explicitly enabled thinking mode. Legacy models that reject 'think' with HTTP 400
        # automatically retry without the parameter in _stream_chat / _non_stream_chat.
        if options and "think" in options:
            payload["think"] = bool(options["think"])
        else:
            payload["think"] = False

        if stream:
            yield from self._stream_chat(payload, model)
        else:
            yield from self._non_stream_chat(payload, model)

    def _classify_chat_error(self, model: str, status_code: int, err_msg: str) -> ChatbotError:
        lowered = err_msg.lower()
        if status_code == 404 or ("not found" in lowered and "model" in lowered):
            return ChatModelMissingError(model, details=err_msg)
        return ModelLoadError(model, details=err_msg)

    def _non_stream_chat(self, payload: dict[str, Any], model: str) -> Iterator[str]:
        with self._translate_http_errors(model=model), self._client() as client:
            resp = client.post("/api/chat", json=payload)
            if resp.status_code == 400 and "think" in payload:
                retry_payload = {k: v for k, v in payload.items() if k != "think"}
                resp = client.post("/api/chat", json=retry_payload)
            if resp.status_code >= 400:
                err = self._extract_error_message(resp)
                raise self._classify_chat_error(model, resp.status_code, err)
            data = resp.json()
            if isinstance(data, dict) and data.get("error"):
                raise self._classify_chat_error(model, 500, str(data["error"]))
            if isinstance(data, dict):
                self._record_chat_stats(data)
            msg = data.get("message", {}) if isinstance(data, dict) else {}
            content = str(msg.get("content") or "")
            if content:
                yield content

    def _stream_chat(self, payload: dict[str, Any], model: str) -> Iterator[str]:
        with self._translate_http_errors(model=model), self._client() as client:
            with client.stream("POST", "/api/chat", json=payload) as resp:
                if resp.status_code == 400 and "think" in payload:
                    # Consume and retry without 'think' parameter
                    resp.read()
                    retry_payload = {k: v for k, v in payload.items() if k != "think"}
                    yield from self._stream_chat(retry_payload, model)
                    return

                if resp.status_code >= 400:
                    resp.read()
                    err = self._extract_error_message(resp)
                    raise self._classify_chat_error(model, resp.status_code, err)

                for line in resp.iter_lines():
                    if not line:
                        continue
                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if not isinstance(chunk, dict):
                        continue
                    if chunk.get("error"):
                        raise self._classify_chat_error(model, 500, str(chunk["error"]))
                    is_done = bool(chunk.get("done"))
                    if is_done:
                        self._record_chat_stats(chunk)
                    msg = chunk.get("message")
                    if isinstance(msg, dict):
                        token = msg.get("content")
                        if token:
                            yield str(token)
                    if is_done:
                        break

    def embed(self, texts: list[str], model: str) -> list[list[float]]:
        """Generate embeddings via Ollama /api/embed in safe batches."""
        if not texts:
            return []

        device = self.embedding_device
        keep_alive = self.keep_alive
        all_vectors: list[list[float]] = []
        with self._translate_http_errors(model=model), self._client() as client:
            for start in range(0, len(texts), EMBED_BATCH_SIZE):
                batch = texts[start : start + EMBED_BATCH_SIZE]
                embed_payload: dict[str, Any] = {
                    "model": model,
                    "input": batch,
                    "keep_alive": keep_alive,
                }
                if device == "cpu":
                    embed_payload["options"] = {"num_gpu": 0}
                resp = client.post(
                    "/api/embed",
                    json=embed_payload,
                )
                if resp.status_code >= 400:
                    err = self._extract_error_message(resp)
                    lowered = err.lower()
                    if resp.status_code == 404 or ("not found" in lowered and "model" in lowered):
                        raise EmbeddingModelMissingError(model, details=err)
                    raise ModelLoadError(model, details=err)

                data = resp.json()
                if isinstance(data, dict) and data.get("error"):
                    err = str(data["error"])
                    if "not found" in err.lower():
                        raise EmbeddingModelMissingError(model, details=err)
                    raise ModelLoadError(model, details=err)

                embeddings = data.get("embeddings") if isinstance(data, dict) else None
                if not isinstance(embeddings, list) or len(embeddings) != len(batch):
                    raise ModelLoadError(
                        model,
                        details=f"Unexpected embedding payload from Ollama for model '{model}'.",
                    )
                all_vectors.extend([[float(v) for v in vec] for vec in embeddings])

        return all_vectors

    def pull_model(self, model: str) -> Iterator[dict[str, Any]]:
        """Stream download progress from Ollama /api/pull (user-initiated only)."""
        pull_timeout = httpx.Timeout(None, connect=5.0)
        cleaned_model = model.strip()
        if not cleaned_model:
            raise ChatbotError(code="validation_error", message="Model name is required.")

        with (
            self._translate_http_errors(offline_on_network_error=True),
            self._client(timeout=pull_timeout) as client,
        ):
            with client.stream(
                "POST",
                "/api/pull",
                json={"model": cleaned_model, "stream": True},
            ) as resp:
                if resp.status_code >= 400:
                    resp.read()
                    err = self._extract_error_message(resp)
                    self._raise_pull_error(cleaned_model, err)

                for line in resp.iter_lines():
                    if not line:
                        continue
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if not isinstance(event, dict):
                        continue
                    if event.get("error"):
                        self._raise_pull_error(cleaned_model, str(event["error"]))

                    total = int(event.get("total") or 0)
                    completed = int(event.get("completed") or 0)
                    percent: float | None = None
                    if total > 0:
                        percent = round(min(100.0, (completed / total) * 100.0), 1)

                    yield {
                        "status": str(event.get("status") or "downloading"),
                        "digest": event.get("digest"),
                        "total": total,
                        "completed": completed,
                        "percent": percent,
                    }

    def _raise_pull_error(self, model: str, err_msg: str) -> None:
        lowered = err_msg.lower()
        if any(indicator in lowered for indicator in NETWORK_ERROR_INDICATORS):
            raise ModelDownloadOfflineError(details=err_msg)
        if "pull model manifest" in lowered and (
            "manifest" not in lowered.split("pull model manifest", 1)[-1]
            or "https://" in lowered
            or "registry.ollama.ai" in lowered
        ):
            if "file does not exist" not in lowered and "not found" not in lowered:
                raise ModelDownloadOfflineError(details=err_msg)
        raise ChatbotError(
            code="pull_failed",
            message=f"Failed to download model '{model}'.",
            details=err_msg,
            status_hint=400,
        )
