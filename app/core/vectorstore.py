"""Persistent ChromaDB vector store with per-profile isolation and metadata safety (§4.2, §4.7)."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import re
from typing import Any

import chromadb
from chromadb.config import Settings

from app.core.backends.base import models_match, normalize_model_tag
from app.core.errors import CollectionMetadataMismatchError
from app.core.ingestion import Chunk, format_location_label


@dataclass(frozen=True)
class CollectionInspection:
    """Summary of a profile's Chroma collection state and stored embedding metadata."""

    exists: bool
    chunk_count: int
    stored_model: str | None
    stored_dimension: int | None


@dataclass(frozen=True)
class RetrievedChunk:
    """A retrieved chunk with normalized similarity score in [0.0, 1.0]."""

    chunk_id: str
    text: str
    score: float
    distance: float
    source_file: str
    chunk_index: int
    file_hash: str
    doc_id: str | None = None
    page: int | None = None
    section: str | None = None

    @property
    def location_label(self) -> str | None:
        return format_location_label(self.page, self.section)

    def to_dict(self, citation_index: int | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "score": round(self.score, 4),
            "source_file": self.source_file,
            "chunk_index": self.chunk_index,
            "page": self.page,
            "section": self.section,
            "location_label": self.location_label,
        }
        if citation_index is not None:
            payload["index"] = citation_index
        return payload


class VectorStore:
    """Manages isolated persistent Chroma collections per bot profile."""

    def __init__(self, persist_dir: Path) -> None:
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=Settings(anonymized_telemetry=False),
        )

    @staticmethod
    def collection_name_for_profile(profile_id: str) -> str:
        """Build a collision-free Chroma-safe collection name (3-63 chars)."""
        safe = re.sub(r"[^a-zA-Z0-9_-]", "_", profile_id).strip("_-") or "default"
        digest = hashlib.sha256(profile_id.encode("utf-8")).hexdigest()[:8]
        return f"prof_{safe[:42]}_{digest}"

    def _get_existing_collection(self, profile_id: str):
        name = self.collection_name_for_profile(profile_id)
        existing_names = {
            c.name if hasattr(c, "name") else str(c)
            for c in self._client.list_collections()
        }
        if name not in existing_names:
            return None
        return self._client.get_collection(name)

    def inspect_collection(self, profile_id: str) -> CollectionInspection:
        """Inspect whether a profile collection exists and read its stored metadata."""
        col = self._get_existing_collection(profile_id)
        if col is None:
            return CollectionInspection(
                exists=False,
                chunk_count=0,
                stored_model=None,
                stored_dimension=None,
            )
        meta = col.metadata or {}
        stored_model = meta.get("embedding_model")
        raw_dim = meta.get("embedding_dimension")
        stored_dim = int(raw_dim) if isinstance(raw_dim, (int, float)) else None
        return CollectionInspection(
            exists=True,
            chunk_count=col.count(),
            stored_model=str(stored_model) if stored_model else None,
            stored_dimension=stored_dim,
        )

    def validate_compatibility(
        self,
        profile_id: str,
        expected_embedding_model: str,
        actual_dimension: int | None = None,
    ) -> CollectionInspection:
        """Verify stored collection metadata matches profile's embedding model and vector dim (§4.7)."""
        info = self.inspect_collection(profile_id)
        if not info.exists:
            return info

        # Empty collection with matching model is clean; mismatched model still blocks if tampered
        model_mismatch = (
            info.stored_model is None
            or not models_match(info.stored_model, expected_embedding_model)
        )
        dim_mismatch = (
            info.stored_dimension is None
            or (actual_dimension is not None and info.stored_dimension != actual_dimension)
        )

        if model_mismatch or dim_mismatch:
            raise CollectionMetadataMismatchError(
                expected_model=expected_embedding_model,
                stored_model=info.stored_model,
                expected_dimension=actual_dimension,
                stored_dimension=info.stored_dimension,
                estimated_chunks=info.chunk_count,
            )
        return info

    def add_chunks(
        self,
        *,
        profile_id: str,
        doc_id: str,
        chunks: list[Chunk],
        embeddings: list[list[float]],
        embedding_model: str,
    ) -> int:
        """Insert embedded document chunks into the profile's isolated Chroma collection."""
        if not chunks:
            return 0
        if len(chunks) != len(embeddings):
            raise ValueError("Number of chunks and embeddings must match.")

        dim = len(embeddings[0])
        self.validate_compatibility(profile_id, embedding_model, actual_dimension=dim)

        col_name = self.collection_name_for_profile(profile_id)
        col = self._get_existing_collection(profile_id)
        if col is None:
            col = self._client.create_collection(
                name=col_name,
                metadata={
                    "hnsw:space": "cosine",
                    "embedding_model": normalize_model_tag(embedding_model),
                    "embedding_dimension": dim,
                },
            )

        ids: list[str] = []
        documents: list[str] = []
        metadatas: list[dict[str, str | int | float | bool]] = []

        for chunk in chunks:
            ids.append(f"{doc_id}_{chunk.chunk_index}")
            documents.append(chunk.text)
            meta: dict[str, str | int | float | bool] = {
                "source_file": chunk.source_file,
                "doc_id": doc_id,
                "file_hash": chunk.file_hash,
                "chunk_index": chunk.chunk_index,
            }
            if chunk.page is not None:
                meta["page"] = int(chunk.page)
            if chunk.section is not None:
                meta["section"] = str(chunk.section)
            metadatas.append(meta)

        col.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )
        return len(chunks)

    def delete_document_chunks(
        self,
        profile_id: str,
        *,
        doc_id: str | None = None,
        file_hash: str | None = None,
    ) -> None:
        """Remove all chunks belonging to a deleted document."""
        col = self._get_existing_collection(profile_id)
        if col is None:
            return
        if doc_id:
            col.delete(where={"doc_id": doc_id})
        elif file_hash:
            col.delete(where={"file_hash": file_hash})

    def reset_collection(self, profile_id: str) -> None:
        """Drop a profile's entire vector collection (used on profile deletion or re-index)."""
        col = self._get_existing_collection(profile_id)
        if col is not None:
            self._client.delete_collection(self.collection_name_for_profile(profile_id))

    def peek_samples(self, profile_id: str, limit: int = 6) -> list[dict[str, Any]]:
        """Return a few representative chunk snippets per document for capability overview queries."""
        col = self._get_existing_collection(profile_id)
        if col is None or col.count() == 0:
            return []
        raw = col.get(limit=max(limit * 3, 12), include=["documents", "metadatas"])
        docs = raw.get("documents") or []
        metas = raw.get("metadatas") or []

        seen_files: set[str] = set()
        samples: list[dict[str, Any]] = []
        for text, meta in zip(docs, metas):
            meta_dict = meta or {}
            src = str(meta_dict.get("source_file") or "Unknown")
            if src not in seen_files:
                seen_files.add(src)
                samples.append(
                    {
                        "source_file": src,
                        "section": meta_dict.get("section"),
                        "page": meta_dict.get("page"),
                        "preview": str(text or "")[:240].strip(),
                    }
                )
                if len(samples) >= limit:
                    break
        return samples

    def query(
        self,
        *,
        profile_id: str,
        query_embedding: list[float],
        expected_embedding_model: str,
        top_k: int = 5,
    ) -> list[RetrievedChunk]:
        """Retrieve top_k chunks and convert cosine distances to [0, 1] similarity scores."""
        info = self.validate_compatibility(
            profile_id,
            expected_embedding_model,
            actual_dimension=len(query_embedding),
        )
        if not info.exists or info.chunk_count == 0:
            return []

        col = self._get_existing_collection(profile_id)
        if col is None:
            return []

        n_results = max(1, min(top_k, info.chunk_count))
        result = col.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )

        ids = (result.get("ids") or [[]])[0]
        docs = (result.get("documents") or [[]])[0]
        metas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]

        retrieved: list[RetrievedChunk] = []
        for cid, text, meta, dist in zip(ids, docs, metas, distances):
            meta_dict = meta or {}
            distance_f = float(dist)
            # Cosine distance d = 1 - cos_sim in [0, 2]; clamp similarity to [0.0, 1.0]
            similarity = max(0.0, min(1.0, 1.0 - distance_f))
            page_val = meta_dict.get("page")
            section_val = meta_dict.get("section")
            retrieved.append(
                RetrievedChunk(
                    chunk_id=str(cid),
                    text=str(text or ""),
                    score=similarity,
                    distance=distance_f,
                    source_file=str(meta_dict.get("source_file") or "Unknown"),
                    chunk_index=int(meta_dict.get("chunk_index") or 0),
                    file_hash=str(meta_dict.get("file_hash") or ""),
                    doc_id=str(meta_dict["doc_id"]) if "doc_id" in meta_dict else None,
                    page=int(page_val) if page_val is not None else None,
                    section=str(section_val) if section_val is not None else None,
                )
            )

        retrieved.sort(key=lambda item: item.score, reverse=True)
        return retrieved
