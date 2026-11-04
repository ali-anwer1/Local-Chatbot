"""Framework-agnostic application service orchestrating profiles, ingestion, saved chats, and RAG."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from app.config import AppConfig, AppSettings
from app.core.availability import ModelAvailabilityManager
from app.core.backends.base import LLMBackend, models_match
from app.core.backends.ollama import OllamaBackend
from app.core.errors import (
    ReindexConfirmationRequiredError,
    ValidationError,
)
from app.core.ingestion import (
    Chunk,
    ParseResult,
    chunk_segments,
    compute_sha256,
    parse_document,
)
from app.core.languages import (
    LANGUAGE_BY_CODE,
    SUPPORTED_LANGUAGES,
    SUPPORTED_LANGUAGE_CODES,
    assemble_translation_messages,
)
from app.core.profiles import DocumentRecord, Profile, ProfileManager
from app.core.rag import RAGEngine
from app.core.vectorstore import VectorStore


class ChatbotService:
    """High-level domain service used by FastAPI routes, CLI scripts, and unit tests."""

    def __init__(
        self,
        config: AppConfig,
        backend: LLMBackend | None = None,
        *,
        app_settings: AppSettings | None = None,
    ) -> None:
        self.config = config
        self.app_settings = (
            app_settings
            or getattr(backend, "app_settings", None)
            or AppSettings.from_config(config)
        )
        if backend is None:
            self.backend: LLMBackend = OllamaBackend(
                base_url=config.ollama_base_url,
                app_settings=self.app_settings,
            )
        else:
            self.backend = backend
        self.profiles = ProfileManager(
            config.data_dir,
            default_chat_model=config.default_chat_model,
            default_embedding_model=config.default_embedding_model,
        )
        self.vectorstore = VectorStore(config.data_dir / "chroma")
        self.availability = ModelAvailabilityManager(self.backend, config)
        self.rag = RAGEngine(
            backend=self.backend,
            vectorstore=self.vectorstore,
            availability=self.availability,
        )

    # --- Health, App Settings & Models ---

    def get_app_settings(self) -> dict[str, Any]:
        """Return current machine-wide runtime settings."""
        return self.app_settings.to_dict()

    def list_languages(self) -> list[dict[str, Any]]:
        """Return the supported languages definition list."""
        return [l.to_dict() for l in SUPPORTED_LANGUAGES]

    def update_app_settings(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Validate and persist machine-wide runtime settings without triggering re-indexing."""
        return self.app_settings.update(payload)

    def startup_check(self) -> dict[str, Any]:
        """Run initial model availability check at startup without crashing (§4.7)."""
        return self.availability.get_system_status(force_refresh=True)

    def get_health(self, *, force_refresh: bool = False) -> dict[str, Any]:
        return self.availability.get_system_status(force_refresh=force_refresh)

    def pull_model(self, model: str) -> Iterator[dict[str, Any]]:
        """Stream user-initiated model download and refresh model cache when complete."""
        try:
            for progress in self.backend.pull_model(model):
                yield progress
        finally:
            self.availability.invalidate_cache()

    # --- Profiles ---

    def _count_chunks(self, profile_id: str, docs: list[DocumentRecord]) -> int:
        inspection = self.vectorstore.inspect_collection(profile_id)
        return (
            inspection.chunk_count
            if inspection.exists
            else sum(d.chunk_count for d in docs)
        )

    def _prepare_chunks(
        self,
        profile: Profile,
        filename: str,
        data: bytes,
        file_hash: str,
        embed_model: str,
    ) -> tuple[ParseResult, list[Chunk], list[list[float]]]:
        parsed = parse_document(data, filename)
        if parsed.is_scanned_pdf:
            return parsed, [], []

        chunks = chunk_segments(
            parsed.segments,
            file_hash=file_hash,
            chunk_size=profile.chunk_size,
            chunk_overlap=profile.chunk_overlap,
        )
        if not chunks:
            return parsed, [], []

        embeddings = self.backend.embed(
            [c.text for c in chunks], model=embed_model
        )
        return parsed, chunks, embeddings

    def _serialize_profile(
        self, profile: Profile, *, force_refresh: bool = False
    ) -> dict[str, Any]:
        data = profile.to_dict()
        docs = self.profiles.list_documents(profile.id)
        data["document_count"] = len(docs)
        data["total_chunks"] = self._count_chunks(profile.id, docs)
        data["saved_chat_count"] = len(self.profiles.list_chats(profile.id))
        data["availability"] = self.availability.evaluate_profile(
            profile, self.vectorstore, force_refresh=force_refresh
        ).to_dict()
        return data

    def list_profiles(self) -> list[dict[str, Any]]:
        all_profiles = self.profiles.list_profiles()
        return [self._serialize_profile(p) for p in all_profiles]

    def get_profile(
        self, profile_id: str, *, force_refresh: bool = False
    ) -> dict[str, Any]:
        profile = self.profiles.get_profile(profile_id)
        return self._serialize_profile(profile, force_refresh=force_refresh)

    def create_profile(self, payload: dict[str, Any]) -> dict[str, Any]:
        profile = self.profiles.create_profile(payload)
        return self._serialize_profile(profile)

    def update_profile(
        self,
        profile_id: str,
        payload: dict[str, Any],
        *,
        confirm_reindex: bool = False,
    ) -> dict[str, Any]:
        """Update profile fields; record settings_history and re-index if embedding_model changes."""
        existing = self.profiles.get_profile(profile_id)
        updated = existing.with_updates(payload)
        embed_changed = not models_match(
            existing.embedding_model, updated.embedding_model
        )

        docs = self.profiles.list_documents(profile_id)
        if embed_changed and not confirm_reindex:
            raise ReindexConfirmationRequiredError(
                current_embedding_model=existing.embedding_model,
                requested_embedding_model=updated.embedding_model,
                estimated_chunks=self._count_chunks(profile_id, docs),
                document_count=len(docs),
            )

        updated.validate()

        reindex_summary: dict[str, Any] | None = None
        if embed_changed:
            reindex_summary = self._perform_reindex(updated, docs)

        saved = self.profiles.save_profile(updated, previous_profile=existing)
        result = self._serialize_profile(saved)
        if reindex_summary is not None:
            result["reindexed"] = reindex_summary
        return result

    def reindex_profile(self, profile_id: str) -> dict[str, Any]:
        """Rebuild a profile's vector collection from its stored documents (§4.7, §7.10)."""
        profile = self.profiles.get_profile(profile_id)
        docs = self.profiles.list_documents(profile_id)
        summary = self._perform_reindex(profile, docs)
        result = self._serialize_profile(profile)
        result["reindexed"] = summary
        return result

    def _perform_reindex(
        self, profile: Profile, docs: list[DocumentRecord]
    ) -> dict[str, Any]:
        """Rebuild the Chroma collection using profile.embedding_model after precomputing all embeddings."""
        effective_embed = self.availability.resolve_embedding_model(profile)

        prepared_batches: list[tuple[DocumentRecord, list[Chunk], list[list[float]]]] = []
        updated_records: list[DocumentRecord] = []
        total_chunks = 0

        for record in docs:
            raw_bytes = self.profiles.read_stored_document_bytes(profile.id, record)
            parsed, chunks, embeddings = self._prepare_chunks(
                profile,
                record.filename,
                raw_bytes,
                record.file_hash,
                effective_embed,
            )
            if parsed.is_scanned_pdf:
                record.status = "scanned_unsupported"
                record.warning = parsed.warning
                record.chunk_count = 0
                updated_records.append(record)
                continue
            if not chunks:
                record.chunk_count = 0
                updated_records.append(record)
                continue

            record.status = "indexed"
            record.warning = None
            record.chunk_count = len(chunks)
            total_chunks += len(chunks)
            prepared_batches.append((record, chunks, embeddings))
            updated_records.append(record)

        # All parsing, chunking, and embedding succeeded above so transient backend
        # failures cannot wipe existing vectors. Not atomic across the Chroma write:
        # if a failure or crash occurs after reset_collection() and before the add_chunks()
        # loop and manifest save complete, the collection is left empty or partial until
        # reindex_profile() is run again.
        self.vectorstore.reset_collection(profile.id)
        for record, chunks, embeddings in prepared_batches:
            self.vectorstore.add_chunks(
                profile_id=profile.id,
                doc_id=record.id,
                chunks=chunks,
                embeddings=embeddings,
                embedding_model=profile.embedding_model,
            )

        self.profiles.save_documents_manifest(profile.id, updated_records)
        return {
            "documents_reindexed": len(prepared_batches),
            "total_chunks": total_chunks,
            "embedding_model": profile.embedding_model,
        }

    def delete_profile(self, profile_id: str) -> None:
        self.profiles.get_profile(profile_id)
        self.vectorstore.reset_collection(profile_id)
        self.profiles.delete_profile(profile_id)

    # --- Documents & Ingestion ---

    def list_documents(self, profile_id: str) -> list[dict[str, Any]]:
        self.profiles.get_profile(profile_id)
        return [d.to_dict() for d in self.profiles.list_documents(profile_id)]

    def ingest_document(
        self,
        profile_id: str,
        *,
        filename: str,
        data: bytes,
    ) -> dict[str, Any]:
        """Parse, deduplicate, chunk, embed, and store a document in the profile's KB (§4.2)."""
        profile = self.profiles.get_profile(profile_id)
        if not filename or not filename.strip():
            raise ValidationError("Filename is required.")

        file_hash = compute_sha256(data)
        existing_docs = self.profiles.list_documents(profile_id)

        for doc in existing_docs:
            if doc.file_hash == file_hash:
                return {
                    "document": doc.to_dict(),
                    "deduplicated": True,
                    "message": f"'{doc.filename}' is already indexed (identical content).",
                }

        effective_embed = self.availability.require_ready_for_ingestion(
            profile, self.vectorstore
        )
        same_name_docs = [
            doc
            for doc in existing_docs
            if doc.filename.lower() == filename.strip().lower()
        ]
        parsed, chunks, embeddings = self._prepare_chunks(
            profile, filename, data, file_hash, effective_embed
        )

        if parsed.is_scanned_pdf:
            record = self.profiles.store_document_file(
                profile_id,
                filename=filename,
                data=data,
                file_hash=file_hash,
                chunk_count=0,
                status="scanned_unsupported",
                warning=parsed.warning,
            )
            for old_doc in same_name_docs:
                self.delete_document(profile_id, old_doc.id)
            return {
                "document": record.to_dict(),
                "deduplicated": False,
                "warning": parsed.warning,
                "message": f"'{record.filename}' flagged as {parsed.warning}; 0 chunks indexed.",
            }

        if not chunks:
            raise ValidationError(
                f"Document '{filename}' contains no extractable text."
            )

        record = self.profiles.store_document_file(
            profile_id,
            filename=filename,
            data=data,
            file_hash=file_hash,
            chunk_count=len(chunks),
            status="indexed",
            warning=None,
        )

        try:
            self.vectorstore.add_chunks(
                profile_id=profile.id,
                doc_id=record.id,
                chunks=chunks,
                embeddings=embeddings,
                embedding_model=profile.embedding_model,
            )
        except Exception:
            self.profiles.remove_document_record(profile.id, record.id)
            raise

        for old_doc in same_name_docs:
            self.delete_document(profile_id, old_doc.id)

        return {
            "document": record.to_dict(),
            "deduplicated": False,
            "warning": None,
            "message": f"Indexed '{record.filename}' ({len(chunks)} chunks).",
        }

    def delete_document(self, profile_id: str, doc_id: str) -> dict[str, Any]:
        """Remove a document and all of its chunks from the vector store (§4.2 step 5)."""
        self.profiles.get_profile(profile_id)
        removed = self.profiles.remove_document_record(profile_id, doc_id)
        self.vectorstore.delete_document_chunks(
            profile_id, doc_id=removed.id, file_hash=removed.file_hash
        )
        return {"deleted": removed.to_dict()}

    # --- Saved Chats ---

    def list_saved_chats(self, profile_id: str) -> list[dict[str, Any]]:
        self.profiles.get_profile(profile_id)
        return self.profiles.list_chats(profile_id)

    def get_saved_chat(
        self,
        profile_id: str,
        chat_id: str,
    ) -> dict[str, Any]:
        return self.profiles.get_chat(profile_id, chat_id).to_dict()

    def save_chat(
        self,
        profile_id: str,
        *,
        turns: list[dict[str, Any]],
        title: str | None = None,
        chat_id: str | None = None,
    ) -> dict[str, Any]:
        saved = self.profiles.save_chat(
            profile_id,
            turns=turns,
            title=title,
            chat_id=chat_id,
        )
        return saved.to_dict()

    def delete_saved_chat(self, profile_id: str, chat_id: str) -> dict[str, Any]:
        self.profiles.get_profile(profile_id)
        self.profiles.delete_chat(profile_id, chat_id)
        return {"deleted": chat_id}

    # --- Chat ---

    def stream_chat(
        self,
        profile_id: str,
        *,
        message: str,
        history: list[dict[str, str]] | None = None,
        runtime_options: dict[str, Any] | None = None,
        requested_language: str | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Stream RAG chat events using client-supplied conversation history."""
        profile = self.profiles.get_profile(profile_id)
        docs_overview = self.list_documents(profile_id)
        yield from self.rag.stream_query(
            profile=profile,
            question=message,
            history=list(history) if history else [],
            runtime_options=runtime_options,
            documents_overview=docs_overview,
            requested_language=requested_language,
        )

    # --- Translation (§ Part 2) ---

    def stream_translate(
        self,
        *,
        text: str,
        target_language: str,
        profile_id: str,
        source_language: str | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Stream translation of an answer using the profile's chat model (§ Part 2).

        No retrieval or refusal gate is invoked. Uses temperature 0.1 and think: false.
        """
        cleaned_text = text.strip()
        if not cleaned_text:
            raise ValidationError("Text to translate cannot be empty.")
        if target_language not in SUPPORTED_LANGUAGE_CODES:
            raise ValidationError(f"Target language '{target_language}' is not supported.")

        profile = self.profiles.get_profile(profile_id)
        chat_model = self.availability.require_ready_for_chat(profile, self.vectorstore)[0]

        target_entry = LANGUAGE_BY_CODE[target_language]
        yield {
            "event": "status",
            "data": {
                "label": f"Translating into {target_entry.native_name}…",
                "target_language": target_language,
            },
        }

        messages = assemble_translation_messages(
            text=cleaned_text,
            target_language=target_language,
            source_language=source_language,
        )

        tokens: list[str] = []
        for token in self.backend.chat(
            messages=messages,
            model=chat_model,
            options={"temperature": 0.1, "think": False},
        ):
            tokens.append(token)
            yield {"event": "token", "data": {"text": token}}

        full_text = "".join(tokens).strip()
        yield {
            "event": "done",
            "data": {
                "full_text": full_text,
                "target_language": target_language,
            },
        }

    def translate(
        self,
        *,
        text: str,
        target_language: str,
        profile_id: str,
        source_language: str | None = None,
    ) -> dict[str, Any]:
        """Synchronous wrapper for stream_translate for tests and CLI."""
        tokens: list[str] = []
        done_data: dict[str, Any] = {}
        for ev in self.stream_translate(
            text=text,
            target_language=target_language,
            profile_id=profile_id,
            source_language=source_language,
        ):
            if ev["event"] == "token":
                tokens.append(ev["data"].get("text", ""))
            elif ev["event"] == "done":
                done_data = ev["data"]
        return {
            "full_text": done_data.get("full_text") or "".join(tokens).strip(),
            "target_language": done_data.get("target_language", target_language),
        }
