"""Thin FastAPI HTTP and SSE layer for Local RAG Chatbot (§3, §4.6, §6)."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import asynccontextmanager
import json
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Query, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import uvicorn

from app.config import load_config
from app.core.errors import ChatbotError, ValidationError
from app.core.service import ChatbotService


STATIC_DIR = Path(__file__).resolve().parent / "static"


class PullModelRequest(BaseModel):
    model: str
    confirmed: bool = False


class ProfilePayload(BaseModel):
    name: str | None = None
    preamble: str | None = None
    chat_model: str | None = None
    embedding_model: str | None = None
    temperature: float | None = None
    top_k: int | None = None
    similarity_threshold: float | None = None
    refusal_message: str | None = None
    chunk_size: int | None = None
    chunk_overlap: int | None = None
    history_turns: int | None = None
    enable_thinking: bool | None = None
    confirm_reindex: bool = False


class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"
    history: list[dict[str, str]] | None = None
    options: dict[str, Any] | None = Field(default=None)


class SaveChatRequest(BaseModel):
    id: str | None = None
    title: str | None = None
    turns: list[dict[str, Any]]


def _format_sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _stream_sse_with_priming(
    events: Iterator[dict[str, Any]],
) -> StreamingResponse:
    """Prime first event before sending HTTP 200 headers so pre-stream errors return structured JSON."""
    first = next(events)

    def generator() -> Iterator[str]:
        yield _format_sse(first["event"], first["data"])
        try:
            for item in events:
                yield _format_sse(item["event"], item["data"])
        except ChatbotError as exc:
            yield _format_sse("error", exc.to_dict())
        except Exception as exc:
            yield _format_sse(
                "error",
                {
                    "code": "stream_error",
                    "message": "An unexpected error occurred during streaming.",
                    "details": str(exc),
                },
            )

    return StreamingResponse(
        generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


def create_app(service: ChatbotService | None = None) -> FastAPI:
    """Create and configure the FastAPI application instance."""
    config = service.config if service is not None else load_config()
    svc = service if service is not None else ChatbotService(config=config)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        svc.startup_check()
        yield

    app = FastAPI(
        title="Local RAG Chatbot",
        version="1.1.0",
        docs_url="/api/docs",
        openapi_url="/api/openapi.json",
        lifespan=lifespan,
    )
    app.state.service = svc

    @app.middleware("http")
    async def offline_security_headers(request: Request, call_next):
        response = await call_next(request)
        if request.url.path == "/" or request.url.path.startswith("/static"):
            response.headers["Cache-Control"] = "no-cache"
            response.headers["Content-Security-Policy"] = (
                "default-src 'self'; "
                "script-src 'self'; "
                "style-src 'self' 'unsafe-inline'; "
                "img-src 'self' data:; "
                "font-src 'self'; "
                "connect-src 'self'; "
                "object-src 'none'; "
                "base-uri 'self'"
            )
        return response

    @app.exception_handler(ChatbotError)
    async def handle_chatbot_error(_request: Request, exc: ChatbotError) -> JSONResponse:
        return JSONResponse(status_code=exc.status_hint, content=exc.to_dict())

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(
        _request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content={
                "code": "validation_error",
                "message": "Invalid request payload.",
                "details": str(exc),
            },
        )

    @app.exception_handler(Exception)
    async def handle_unexpected_error(_request: Request, exc: Exception) -> JSONResponse:
        return JSONResponse(
            status_code=500,
            content={
                "code": "internal_error",
                "message": "An unexpected internal error occurred.",
                "details": str(exc),
            },
        )

    # --- UI Static Serving ---

    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.get("/", include_in_schema=False)
    def serve_index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    # --- API Routes (§4.6) ---

    @app.get("/api/health")
    def get_health(refresh: bool = Query(default=False)) -> dict[str, Any]:
        return svc.get_health(force_refresh=refresh)

    @app.get("/api/models")
    def get_models(refresh: bool = Query(default=False)) -> dict[str, Any]:
        return svc.list_models(force_refresh=refresh)

    @app.post("/api/models/pull")
    def pull_model(body: PullModelRequest) -> StreamingResponse:
        if not body.confirmed:
            raise ValidationError(
                "Model downloads must be explicitly confirmed by the user."
            )

        def pull_events() -> Iterator[dict[str, Any]]:
            for progress in svc.pull_model(body.model):
                yield {"event": "progress", "data": progress}
            yield {
                "event": "done",
                "data": {"model": body.model, "status": "success"},
            }

        return _stream_sse_with_priming(pull_events())

    @app.get("/api/profiles")
    def list_profiles() -> dict[str, Any]:
        return {"profiles": svc.list_profiles()}

    @app.post("/api/profiles", status_code=201)
    def create_profile(body: ProfilePayload) -> dict[str, Any]:
        payload = {k: v for k, v in body.model_dump().items() if v is not None}
        return svc.create_profile(payload)

    @app.get("/api/profiles/{profile_id}")
    def get_profile(
        profile_id: str, refresh: bool = Query(default=False)
    ) -> dict[str, Any]:
        return svc.get_profile(profile_id, force_refresh=refresh)

    @app.put("/api/profiles/{profile_id}")
    def update_profile(profile_id: str, body: ProfilePayload) -> dict[str, Any]:
        raw = body.model_dump()
        confirm_reindex = bool(raw.pop("confirm_reindex", False))
        payload = {k: v for k, v in raw.items() if v is not None}
        return svc.update_profile(
            profile_id, payload, confirm_reindex=confirm_reindex
        )

    @app.post("/api/profiles/{profile_id}/reindex")
    def reindex_profile(profile_id: str) -> dict[str, Any]:
        return svc.reindex_profile(profile_id)

    @app.delete("/api/profiles/{profile_id}")
    def delete_profile(profile_id: str) -> dict[str, Any]:
        svc.delete_profile(profile_id)
        return {"deleted": profile_id}

    @app.get("/api/profiles/{profile_id}/documents")
    def list_documents(profile_id: str) -> dict[str, Any]:
        return {"documents": svc.list_documents(profile_id)}

    @app.post("/api/profiles/{profile_id}/documents", status_code=201)
    async def upload_document(
        profile_id: str, file: UploadFile = File(...)
    ) -> dict[str, Any]:
        raw_bytes = await file.read()
        return svc.ingest_document(
            profile_id,
            filename=file.filename or "uploaded_file",
            data=raw_bytes,
        )

    @app.delete("/api/profiles/{profile_id}/documents/{doc_id}")
    def delete_document(profile_id: str, doc_id: str) -> dict[str, Any]:
        return svc.delete_document(profile_id, doc_id)

    # --- Saved Chats Routes ---

    @app.get("/api/profiles/{profile_id}/chats")
    def list_saved_chats(profile_id: str) -> dict[str, Any]:
        return {"chats": svc.list_saved_chats(profile_id)}

    @app.post("/api/profiles/{profile_id}/chats", status_code=201)
    def save_chat(profile_id: str, body: SaveChatRequest) -> dict[str, Any]:
        return svc.save_chat(
            profile_id,
            turns=body.turns,
            title=body.title,
            chat_id=body.id,
        )

    @app.get("/api/profiles/{profile_id}/chats/{chat_id}")
    def get_saved_chat(
        profile_id: str,
        chat_id: str,
        session_id: str | None = Query(default=None),
    ) -> dict[str, Any]:
        del session_id  # Accepted for backward compatibility; ignored
        return svc.get_saved_chat(profile_id, chat_id)

    @app.delete("/api/profiles/{profile_id}/chats/{chat_id}")
    def delete_saved_chat(profile_id: str, chat_id: str) -> dict[str, Any]:
        return svc.delete_saved_chat(profile_id, chat_id)

    # --- Streaming Chat & Session History ---

    @app.post("/api/profiles/{profile_id}/chat")
    def chat(profile_id: str, body: ChatRequest) -> StreamingResponse:
        events = svc.stream_chat(
            profile_id,
            message=body.message,
            history=body.history,
            runtime_options=body.options,
        )
        return _stream_sse_with_priming(events)

    @app.delete("/api/profiles/{profile_id}/history")
    def clear_history(
        profile_id: str, session_id: str = Query(default="default")
    ) -> dict[str, Any]:
        svc.clear_history(profile_id, session_id)
        return {"cleared": True, "profile_id": profile_id, "session_id": session_id}

    return app


class _LazyASGIApp:
    """Lazily instantiates the default FastAPI app on first ASGI request."""

    def __init__(self) -> None:
        self._instance: FastAPI | None = None

    def _get_app(self) -> FastAPI:
        if self._instance is None:
            self._instance = create_app()
        return self._instance

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        target = self._get_app()
        await target(scope, receive, send)


app = _LazyASGIApp()


if __name__ == "__main__":
    cfg = load_config()
    uvicorn.run(
        "app.main:create_app",
        factory=True,
        host="127.0.0.1",
        port=cfg.port,
        reload=False,
    )
