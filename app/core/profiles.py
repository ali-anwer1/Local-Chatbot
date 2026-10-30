"""Bot profile, document manifest, settings history, and saved chat persistence on disk (§4.1)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any
import uuid

from app.core.errors import (
    ChatNotFoundError,
    DocumentNotFoundError,
    ProfileNotFoundError,
    ValidationError,
)


logger = logging.getLogger(__name__)

DEFAULT_REFUSAL_MESSAGE = "I can't answer that from my knowledge base."
MAX_SETTINGS_HISTORY_ENTRIES = 10


def _atomic_write_json(path: Path, data: Any) -> None:
    """Write JSON atomically via a temp file in the same directory and os.replace."""
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


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug[:48] or "bot-profile"


def _sanitize_filename(filename: str) -> str:
    base = Path(filename).name.strip()
    safe = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", base)
    if not safe or safe in {".", ".."}:
        raise ValidationError("Invalid document filename.")
    return safe


def _coerce_stripped_str(val: Any) -> str:
    return "" if val is None else str(val).strip()


def _coerce_raw_str(val: Any) -> str:
    return "" if val is None else str(val)


_PROFILE_MUTABLE_COERCERS: dict[str, Any] = {
    "name": _coerce_stripped_str,
    "preamble": _coerce_raw_str,
    "chat_model": _coerce_stripped_str,
    "embedding_model": _coerce_stripped_str,
    "temperature": float,
    "top_k": int,
    "similarity_threshold": float,
    "refusal_message": _coerce_stripped_str,
    "chunk_size": int,
    "chunk_overlap": int,
    "history_turns": int,
    "enable_thinking": bool,
}


def _drop_empty_model_overrides(data: dict[str, Any]) -> dict[str, Any]:
    return {
        k: v
        for k, v in data.items()
        if not (k in {"chat_model", "embedding_model"} and not str(v or "").strip())
    }


@dataclass
class Profile:
    """Bot profile configuration (§4.1)."""

    id: str
    name: str = "Untitled Bot"
    preamble: str = ""
    chat_model: str = "qwen3.5:4b"
    embedding_model: str = "nomic-embed-text"
    temperature: float = 0.2
    top_k: int = 5
    similarity_threshold: float = 0.5
    refusal_message: str = DEFAULT_REFUSAL_MESSAGE
    chunk_size: int = 1000
    chunk_overlap: int = 150
    history_turns: int = 4
    enable_thinking: bool = False
    settings_history: list[dict[str, Any]] = field(default_factory=list)
    created_at: str = ""
    updated_at: str = ""

    def validate(self) -> None:
        if not self.name or not self.name.strip():
            raise ValidationError("Profile name cannot be empty.")
        if not self.chat_model or not self.chat_model.strip():
            raise ValidationError("Chat model cannot be empty.")
        if not self.embedding_model or not self.embedding_model.strip():
            raise ValidationError("Embedding model cannot be empty.")
        if not (0.0 <= self.temperature <= 2.0):
            raise ValidationError("Temperature must be between 0.0 and 2.0.")
        if not (1 <= self.top_k <= 25):
            raise ValidationError("top_k must be between 1 and 25.")
        if not (0.0 <= self.similarity_threshold <= 1.0):
            raise ValidationError("similarity_threshold must be between 0.0 and 1.0.")
        if not (100 <= self.chunk_size <= 8000):
            raise ValidationError("chunk_size must be between 100 and 8000 characters.")
        if not (0 <= self.chunk_overlap < self.chunk_size):
            raise ValidationError("chunk_overlap must be >= 0 and strictly less than chunk_size.")
        if not (0 <= self.history_turns <= 20):
            raise ValidationError("history_turns must be between 0 and 20.")
        if not self.refusal_message or not self.refusal_message.strip():
            self.refusal_message = DEFAULT_REFUSAL_MESSAGE

    def with_updates(self, payload: dict[str, Any]) -> Profile:
        """Return a copy with coerced mutable fields updated, ignoring unknown/immutable keys."""
        updates = {
            key: _PROFILE_MUTABLE_COERCERS[key](val)
            for key, val in payload.items()
            if key in _PROFILE_MUTABLE_COERCERS and val is not None
        }
        return replace(
            self,
            settings_history=list(self.settings_history),
            **updates,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        *,
        default_chat_model: str = "qwen3.5:4b",
        default_embedding_model: str = "nomic-embed-text",
    ) -> Profile:
        now = _utc_now_iso()
        raw_hist = data.get("settings_history")
        hist = [item for item in raw_hist if isinstance(item, dict)] if isinstance(raw_hist, list) else []
        base = cls(
            id=str(data["id"]),
            chat_model=default_chat_model,
            embedding_model=default_embedding_model,
            settings_history=hist[:MAX_SETTINGS_HISTORY_ENTRIES],
            created_at=str(data.get("created_at") or now),
            updated_at=str(data.get("updated_at") or now),
        )
        profile = base.with_updates(_drop_empty_model_overrides(data))
        profile.validate()
        return profile


def _fmt_arrow(label: str, old: Any, new: Any) -> str:
    return f"{label} ({old} → {new})"


def _fmt_float(label: str, old: Any, new: Any) -> str:
    return f"{label} ({old:.2f} → {new:.2f})"


def _fmt_label_only(label: str, _old: Any, _new: Any) -> str:
    return label


def _fmt_on_off(label: str, _old: Any, new: Any) -> str:
    return f"{label} ({'On' if new else 'Off'})"


_PROFILE_CHANGE_SPECS: list[tuple[str, str, Any]] = [
    ("name", "Name", _fmt_arrow),
    ("preamble", "System Preamble", _fmt_label_only),
    ("chat_model", "Chat Model", _fmt_arrow),
    ("embedding_model", "Embedding Model", _fmt_arrow),
    ("similarity_threshold", "Threshold", _fmt_float),
    ("temperature", "Temperature", _fmt_float),
    ("top_k", "Top-K", _fmt_arrow),
    ("chunk_size", "Chunk Size", _fmt_arrow),
    ("chunk_overlap", "Chunk Overlap", _fmt_arrow),
    ("history_turns", "History Turns", _fmt_arrow),
    ("refusal_message", "Refusal Message", _fmt_label_only),
    ("enable_thinking", "Thinking Mode", _fmt_on_off),
]


def describe_profile_changes(before: Profile, after: Profile) -> list[str]:
    """Build concise, human-readable descriptions of changed settings."""
    changes: list[str] = []
    for field_name, label, formatter in _PROFILE_CHANGE_SPECS:
        old_val = getattr(before, field_name)
        new_val = getattr(after, field_name)
        changed = (
            round(old_val, 3) != round(new_val, 3)
            if isinstance(old_val, float)
            else old_val != new_val
        )
        if changed:
            changes.append(formatter(label, old_val, new_val))
    return changes


@dataclass
class DocumentRecord:
    """Metadata record for an uploaded document inside a profile."""

    id: str
    filename: str
    stored_filename: str
    file_hash: str
    size_bytes: int
    chunk_count: int
    status: str  # "indexed" | "scanned_unsupported"
    warning: str | None = None
    uploaded_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DocumentRecord:
        return cls(
            id=str(data["id"]),
            filename=str(data["filename"]),
            stored_filename=str(data["stored_filename"]),
            file_hash=str(data["file_hash"]),
            size_bytes=int(data.get("size_bytes", 0)),
            chunk_count=int(data.get("chunk_count", 0)),
            status=str(data.get("status", "indexed")),
            warning=data.get("warning"),
            uploaded_at=str(data.get("uploaded_at") or _utc_now_iso()),
        )


@dataclass
class SavedChat:
    """User-saved conversation transcript persisted per profile."""

    id: str
    profile_id: str
    title: str
    turns: list[dict[str, Any]]
    created_at: str = ""
    updated_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_summary_dict(self) -> dict[str, Any]:
        user_turns = sum(1 for t in self.turns if t.get("role") == "user")
        return {
            "id": self.id,
            "profile_id": self.profile_id,
            "title": self.title,
            "turn_count": user_turns,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any], profile_id: str) -> SavedChat:
        now = _utc_now_iso()
        raw_turns = data.get("turns")
        turns = [t for t in raw_turns if isinstance(t, dict)] if isinstance(raw_turns, list) else []
        return cls(
            id=str(data["id"]),
            profile_id=profile_id,
            title=str(data.get("title") or "Saved Conversation").strip()[:120],
            turns=turns,
            created_at=str(data.get("created_at") or now),
            updated_at=str(data.get("updated_at") or now),
        )


class ProfileManager:
    """Manages profile JSON files, uploaded documents, and saved chats under data/profiles/."""

    def __init__(
        self,
        data_dir: Path,
        *,
        default_chat_model: str = "qwen3.5:4b",
        default_embedding_model: str = "nomic-embed-text",
    ) -> None:
        self.data_dir = Path(data_dir)
        self.profiles_dir = self.data_dir / "profiles"
        self.profiles_dir.mkdir(parents=True, exist_ok=True)
        self.default_chat_model = default_chat_model
        self.default_embedding_model = default_embedding_model

    def _validate_profile_id(self, profile_id: str) -> Path:
        if not re.match(r"^[a-z0-9][a-z0-9_-]{0,62}$", profile_id):
            raise ProfileNotFoundError(profile_id)
        folder = (self.profiles_dir / profile_id).resolve()
        if folder.parent != self.profiles_dir.resolve():
            raise ProfileNotFoundError(profile_id)
        return folder

    def profile_dir(self, profile_id: str) -> Path:
        folder = self._validate_profile_id(profile_id)
        if not (folder / "profile.json").exists():
            raise ProfileNotFoundError(profile_id)
        return folder

    def docs_dir(self, profile_id: str) -> Path:
        d = self.profile_dir(profile_id) / "docs"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def chats_dir(self, profile_id: str) -> Path:
        d = self.profile_dir(profile_id) / "chats"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _generate_unique_id(self, name: str) -> str:
        base_slug = _slugify(name)
        candidate = base_slug
        counter = 2
        while (self.profiles_dir / candidate).exists():
            candidate = f"{base_slug}-{counter}"
            counter += 1
        return candidate

    def list_profiles(self) -> list[Profile]:
        profiles: list[Profile] = []
        if not self.profiles_dir.exists():
            return profiles
        for entry in sorted(self.profiles_dir.iterdir()):
            profile_file = entry / "profile.json"
            if entry.is_dir() and profile_file.exists():
                try:
                    with profile_file.open("r", encoding="utf-8") as fh:
                        raw = json.load(fh)
                    profiles.append(
                        Profile.from_dict(
                            raw,
                            default_chat_model=self.default_chat_model,
                            default_embedding_model=self.default_embedding_model,
                        )
                    )
                except Exception as exc:
                    logger.warning(
                        "Skipping unreadable profile at %s: %s", profile_file, exc
                    )
                    continue
        profiles.sort(key=lambda p: p.created_at)
        return profiles

    def get_profile(self, profile_id: str) -> Profile:
        folder = self.profile_dir(profile_id)
        with (folder / "profile.json").open("r", encoding="utf-8") as fh:
            raw = json.load(fh)
        return Profile.from_dict(
            raw,
            default_chat_model=self.default_chat_model,
            default_embedding_model=self.default_embedding_model,
        )

    def create_profile(self, payload: dict[str, Any]) -> Profile:
        name = str(payload.get("name") or "").strip()
        if not name:
            raise ValidationError("Profile name is required.")

        profile_id = self._generate_unique_id(name)
        now = _utc_now_iso()
        base = Profile(
            id=profile_id,
            name=name,
            chat_model=self.default_chat_model,
            embedding_model=self.default_embedding_model,
            created_at=now,
            updated_at=now,
        )
        profile = base.with_updates(_drop_empty_model_overrides(payload))
        profile.validate()

        folder = self.profiles_dir / profile_id
        (folder / "docs").mkdir(parents=True, exist_ok=True)
        (folder / "chats").mkdir(parents=True, exist_ok=True)
        self._write_profile(folder, profile)
        self._write_documents(folder, [])
        return profile

    def save_profile(
        self,
        profile: Profile,
        *,
        previous_profile: Profile | None = None,
    ) -> Profile:
        profile.validate()
        now = _utc_now_iso()
        profile.updated_at = now
        if previous_profile is not None:
            changes = describe_profile_changes(previous_profile, profile)
            entry = {
                "saved_at": now,
                "changes": changes if changes else ["No parameter changes"],
            }
            history = [entry] + list(previous_profile.settings_history)
            profile.settings_history = history[:MAX_SETTINGS_HISTORY_ENTRIES]
        folder = self.profile_dir(profile.id)
        self._write_profile(folder, profile)
        return profile

    def delete_profile(self, profile_id: str) -> None:
        folder = self.profile_dir(profile_id)
        shutil.rmtree(folder)

    def _write_profile(self, folder: Path, profile: Profile) -> None:
        target = folder / "profile.json"
        _atomic_write_json(target, profile.to_dict())

    # --- Document Manifest Helpers ---

    def _documents_manifest_path(self, profile_id: str) -> Path:
        return self.profile_dir(profile_id) / "documents.json"

    def _write_documents(self, folder: Path, docs: list[DocumentRecord]) -> None:
        manifest = folder / "documents.json"
        _atomic_write_json(manifest, [d.to_dict() for d in docs])

    def list_documents(self, profile_id: str) -> list[DocumentRecord]:
        manifest = self._documents_manifest_path(profile_id)
        if not manifest.exists():
            return []
        with manifest.open("r", encoding="utf-8") as fh:
            raw_list = json.load(fh)
        if not isinstance(raw_list, list):
            return []
        return [DocumentRecord.from_dict(item) for item in raw_list if isinstance(item, dict)]

    def get_document(self, profile_id: str, doc_id: str) -> DocumentRecord:
        for doc in self.list_documents(profile_id):
            if doc.id == doc_id:
                return doc
        raise DocumentNotFoundError(doc_id, profile_id)

    def save_documents_manifest(self, profile_id: str, docs: list[DocumentRecord]) -> None:
        folder = self.profile_dir(profile_id)
        self._write_documents(folder, docs)

    def store_document_file(
        self,
        profile_id: str,
        *,
        filename: str,
        data: bytes,
        file_hash: str,
        chunk_count: int,
        status: str,
        warning: str | None = None,
    ) -> DocumentRecord:
        safe_name = _sanitize_filename(filename)
        doc_id = uuid.uuid4().hex[:12]
        stored_name = f"{doc_id}_{safe_name}"
        target_path = self.docs_dir(profile_id) / stored_name
        target_path.write_bytes(data)

        record = DocumentRecord(
            id=doc_id,
            filename=safe_name,
            stored_filename=stored_name,
            file_hash=file_hash,
            size_bytes=len(data),
            chunk_count=chunk_count,
            status=status,
            warning=warning,
            uploaded_at=_utc_now_iso(),
        )
        docs = self.list_documents(profile_id)
        docs.append(record)
        self.save_documents_manifest(profile_id, docs)
        return record

    def read_stored_document_bytes(self, profile_id: str, record: DocumentRecord) -> bytes:
        file_path = self.docs_dir(profile_id) / record.stored_filename
        return file_path.read_bytes()

    def remove_document_record(self, profile_id: str, doc_id: str) -> DocumentRecord:
        docs = self.list_documents(profile_id)
        target: DocumentRecord | None = None
        remaining: list[DocumentRecord] = []
        for doc in docs:
            if doc.id == doc_id:
                target = doc
            else:
                remaining.append(doc)
        if target is None:
            raise DocumentNotFoundError(doc_id, profile_id)

        stored_file = self.docs_dir(profile_id) / target.stored_filename
        if stored_file.exists():
            stored_file.unlink()
        self.save_documents_manifest(profile_id, remaining)
        return target

    # --- Saved Chats Helpers ---

    def _validate_chat_id(self, profile_id: str, chat_id: str) -> Path:
        if not re.match(r"^[a-z0-9][a-z0-9_-]{0,48}$", chat_id):
            raise ChatNotFoundError(chat_id, profile_id)
        cdir = self.chats_dir(profile_id).resolve()
        target = (cdir / f"{chat_id}.json").resolve()
        if target.parent != cdir:
            raise ChatNotFoundError(chat_id, profile_id)
        return target

    def list_chats(self, profile_id: str) -> list[dict[str, Any]]:
        cdir = self.chats_dir(profile_id)
        chats: list[SavedChat] = []
        for path in cdir.glob("*.json"):
            try:
                with path.open("r", encoding="utf-8") as fh:
                    raw = json.load(fh)
                if isinstance(raw, dict):
                    chats.append(SavedChat.from_dict(raw, profile_id))
            except Exception:
                continue
        chats.sort(key=lambda c: c.updated_at, reverse=True)
        return [c.to_summary_dict() for c in chats]

    def get_chat(self, profile_id: str, chat_id: str) -> SavedChat:
        target = self._validate_chat_id(profile_id, chat_id)
        if not target.exists():
            raise ChatNotFoundError(chat_id, profile_id)
        with target.open("r", encoding="utf-8") as fh:
            raw = json.load(fh)
        return SavedChat.from_dict(raw, profile_id)

    def save_chat(
        self,
        profile_id: str,
        *,
        turns: list[dict[str, Any]],
        title: str | None = None,
        chat_id: str | None = None,
    ) -> SavedChat:
        self.get_profile(profile_id)
        if not turns:
            raise ValidationError("Cannot save an empty conversation.")

        now = _utc_now_iso()
        existing_created = now
        if chat_id:
            target = self._validate_chat_id(profile_id, chat_id)
            if target.exists():
                try:
                    old = json.loads(target.read_text(encoding="utf-8"))
                    existing_created = str(old.get("created_at") or now)
                    if not title:
                        title = str(old.get("title") or "")
                except Exception:
                    pass
            resolved_id = chat_id
        else:
            resolved_id = f"chat-{uuid.uuid4().hex[:10]}"
            target = self._validate_chat_id(profile_id, resolved_id)

        resolved_title = (title or "").strip()
        if not resolved_title:
            first_user = next(
                (str(t.get("text") or t.get("content") or "").strip() for t in turns if t.get("role") == "user"),
                "Saved Conversation",
            )
            resolved_title = (
                first_user[:56].rstrip() + ("…" if len(first_user) > 56 else "")
            ) or "Saved Conversation"

        saved = SavedChat(
            id=resolved_id,
            profile_id=profile_id,
            title=resolved_title[:120],
            turns=turns,
            created_at=existing_created,
            updated_at=now,
        )
        _atomic_write_json(target, saved.to_dict())
        return saved

    def delete_chat(self, profile_id: str, chat_id: str) -> None:
        target = self._validate_chat_id(profile_id, chat_id)
        if not target.exists():
            raise ChatNotFoundError(chat_id, profile_id)
        target.unlink()
