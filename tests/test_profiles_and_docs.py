"""Unit tests for profile CRUD, KB isolation, deduplication, deletion, and re-indexing (§7.1, §7.4, §7.7, §7.10)."""

from __future__ import annotations

import io
import logging
from pathlib import Path
from pypdf import PdfWriter
import pytest

from app.core.errors import (
    ChatNotFoundError,
    ModelLoadError,
    ProfileNotFoundError,
    ReindexConfirmationRequiredError,
)
from app.config import atomic_write_json
from app.core.ingestion import SCANNED_PDF_WARNING
from app.core.profiles import Profile, describe_profile_changes
from app.core.service import ChatbotService
from tests.conftest import DeterministicMockBackend


def test_profile_crud_lifecycle(chatbot_service: ChatbotService) -> None:
    created = chatbot_service.create_profile(
        {
            "name": "Engineering Spec Bot",
            "preamble": "Answer strictly from specs.",
            "temperature": 0.15,
            "top_k": 4,
            "similarity_threshold": 0.55,
        }
    )
    pid = created["id"]
    assert created["name"] == "Engineering Spec Bot"
    assert created["temperature"] == 0.15
    assert created["top_k"] == 4
    assert created["similarity_threshold"] == 0.55

    fetched = chatbot_service.get_profile(pid)
    assert fetched["id"] == pid

    updated = chatbot_service.update_profile(
        pid, {"name": "Updated Spec Bot", "similarity_threshold": 0.60}
    )
    assert updated["name"] == "Updated Spec Bot"
    assert updated["similarity_threshold"] == 0.60

    all_profiles = chatbot_service.list_profiles()
    assert len(all_profiles) == 1

    chatbot_service.delete_profile(pid)
    assert chatbot_service.list_profiles() == []
    with pytest.raises(ProfileNotFoundError):
        chatbot_service.get_profile(pid)


def test_document_deduplication_and_deletion_removes_chunks(
    chatbot_service: ChatbotService,
) -> None:
    prof = chatbot_service.create_profile({"name": "KB Test"})
    pid = prof["id"]

    content = b"# Warranty\n\nThe standard hardware warranty is 36-month return-to-depot coverage."
    res1 = chatbot_service.ingest_document(pid, filename="warranty.md", data=content)
    assert res1["deduplicated"] is False
    doc_id = res1["document"]["id"]
    assert res1["document"]["chunk_count"] == 1

    # Re-uploading identical content (same SHA-256) is a no-op (§4.2 step 5)
    res2 = chatbot_service.ingest_document(pid, filename="warranty_copy.md", data=content)
    assert res2["deduplicated"] is True
    assert len(chatbot_service.list_documents(pid)) == 1

    # Query succeeds while document is present
    profile_obj = chatbot_service.profiles.get_profile(pid)
    ans_before = chatbot_service.rag.query(
        profile=profile_obj, question="What is the 36-month warranty?"
    )
    assert ans_before.refused is False
    assert ans_before.sources[0]["source_file"] == "warranty.md"

    # Deleting the document removes its chunks from Chroma (§4.2 step 5, §7.7)
    chatbot_service.delete_document(pid, doc_id)
    assert chatbot_service.list_documents(pid) == []
    assert chatbot_service.vectorstore.inspect_collection(pid).chunk_count == 0

    ans_after = chatbot_service.rag.query(
        profile=profile_obj, question="What is the 36-month warranty?"
    )
    assert ans_after.refused is True


def test_scanned_pdf_flagged_in_document_list(chatbot_service: ChatbotService) -> None:
    prof = chatbot_service.create_profile({"name": "Scan Check"})
    pid = prof["id"]

    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)
    buf = io.BytesIO()
    writer.write(buf)

    res = chatbot_service.ingest_document(pid, filename="scan.pdf", data=buf.getvalue())
    assert res["warning"] == SCANNED_PDF_WARNING
    assert res["document"]["status"] == "scanned_unsupported"
    assert res["document"]["chunk_count"] == 0


def test_strict_profile_isolation(chatbot_service: ChatbotService) -> None:
    """Profile A must never retrieve Profile B's chunks (§7.4)."""
    prof_a = chatbot_service.create_profile({"name": "Profile Alpha"})
    prof_b = chatbot_service.create_profile({"name": "Profile Beta"})

    chatbot_service.ingest_document(
        prof_a["id"],
        filename="alpha.txt",
        data=b"Project-Alpha uses code name Cobalt secret architecture.",
    )
    chatbot_service.ingest_document(
        prof_b["id"],
        filename="beta.txt",
        data=b"Project-Beta uses code name Crimson confidential protocol.",
    )

    p_a = chatbot_service.profiles.get_profile(prof_a["id"])
    p_b = chatbot_service.profiles.get_profile(prof_b["id"])

    # Profile A answers Alpha questions, refuses Beta questions
    res_a_own = chatbot_service.rag.query(profile=p_a, question="What is Project-Alpha Cobalt?")
    assert res_a_own.refused is False
    assert res_a_own.sources[0]["source_file"] == "alpha.txt"

    res_a_cross = chatbot_service.rag.query(
        profile=p_a, question="What is Project-Beta Crimson confidential?"
    )
    assert res_a_cross.refused is True
    assert all(c["source_file"] != "beta.txt" for c in res_a_cross.debug_chunks)

    # Profile B answers Beta questions, refuses Alpha questions
    res_b_own = chatbot_service.rag.query(profile=p_b, question="What is Project-Beta Crimson?")
    assert res_b_own.refused is False
    assert res_b_own.sources[0]["source_file"] == "beta.txt"


def test_changing_embedding_model_requires_confirmation_and_reindexes(
    chatbot_service: ChatbotService,
    mock_backend: DeterministicMockBackend,
) -> None:
    """Changing embedding_model requires confirmation and performs full re-index (§7.10)."""
    prof = chatbot_service.create_profile(
        {"name": "Reindex Bot", "embedding_model": "nomic-embed-text"}
    )
    pid = prof["id"]

    chatbot_service.ingest_document(
        pid,
        filename="power.txt",
        data=b"Input voltage is 12 V to 48 V DC with 34 W peak electrical power.",
    )
    old_inspection = chatbot_service.vectorstore.inspect_collection(pid)
    # nomic-embed-text in mock produces len(VOCAB_TOPICS)+1 dims
    old_dim = old_inspection.stored_dimension
    assert old_dim is not None

    # 1. Updating embedding_model without confirm_reindex=True raises confirmation error & changes nothing
    with pytest.raises(ReindexConfirmationRequiredError) as exc_info:
        chatbot_service.update_profile(
            pid, {"embedding_model": "mxbai-embed-large"}, confirm_reindex=False
        )
    assert exc_info.value.extra["estimated_chunks"] == 1
    assert chatbot_service.get_profile(pid)["embedding_model"] == "nomic-embed-text"

    # 2. Confirming re-index rebuilds Chroma collection with new model & new vector dimension
    updated = chatbot_service.update_profile(
        pid, {"embedding_model": "mxbai-embed-large"}, confirm_reindex=True
    )
    assert updated["embedding_model"] == "mxbai-embed-large"
    assert updated["reindexed"]["total_chunks"] == 1

    new_inspection = chatbot_service.vectorstore.inspect_collection(pid)
    assert new_inspection.stored_dimension == old_dim + 4

    # 3. Retrieval succeeds cleanly after re-indexing!
    p_obj = chatbot_service.profiles.get_profile(pid)
    ans = chatbot_service.rag.query(
        profile=p_obj, question="What is the DC input voltage and peak power?"
    )
    assert ans.refused is False
    assert ans.sources[0]["source_file"] == "power.txt"


def test_settings_save_history_records_latest_changes_and_timestamps(
    chatbot_service: ChatbotService,
) -> None:
    """Updating profile settings records timestamped change descriptions in settings_history."""
    prof = chatbot_service.create_profile({"name": "History Bot", "similarity_threshold": 0.50})
    pid = prof["id"]
    assert prof["settings_history"] == []

    first_save = chatbot_service.update_profile(
        pid, {"similarity_threshold": 0.42, "enable_thinking": True}
    )
    assert len(first_save["settings_history"]) == 1
    latest = first_save["settings_history"][0]
    assert latest["saved_at"]
    assert "Threshold (0.50 → 0.42)" in latest["changes"]
    assert "Thinking Mode (On)" in latest["changes"]

    second_save = chatbot_service.update_profile(pid, {"top_k": 8})
    assert len(second_save["settings_history"]) == 2
    assert "Top-K (5 → 8)" in second_save["settings_history"][0]["changes"]


def test_saved_chats_persistence_and_session_restoration(
    chatbot_service: ChatbotService,
    mock_backend: DeterministicMockBackend,
) -> None:
    """Saved chats persist per profile on disk, and client-supplied history reaches the prompt."""
    prof = chatbot_service.create_profile({"name": "Saved Chat Bot"})
    pid = prof["id"]

    turns = [
        {"role": "user", "text": "What is the DC voltage range?"},
        {
            "role": "assistant",
            "text": "The input voltage range is 12 V to 48 V DC [1].",
            "sources": [{"index": 1, "source_file": "datasheet.pdf"}],
        },
    ]
    saved = chatbot_service.save_chat(pid, turns=turns, title="Voltage Range Chat")
    cid = saved["id"]
    assert saved["title"] == "Voltage Range Chat"

    listed = chatbot_service.list_saved_chats(pid)
    assert len(listed) == 1
    assert listed[0]["id"] == cid
    assert listed[0]["turn_count"] == 1

    # Updating an existing saved chat preserves its ID and updates turn count
    turns_v2 = turns + [
        {"role": "user", "text": "And peak wattage?"},
        {"role": "assistant", "text": "Peak power draw is 34 W [1]."},
    ]
    updated = chatbot_service.save_chat(pid, turns=turns_v2, chat_id=cid)
    assert updated["id"] == cid
    assert chatbot_service.list_saved_chats(pid)[0]["turn_count"] == 2

    # Loading a saved chat has no server-side side effects; passing its turns as history reaches the LLM prompt
    loaded = chatbot_service.get_saved_chat(pid, cid)
    assert len(loaded["turns"]) == 4

    chatbot_service.ingest_document(
        pid,
        filename="power.txt",
        data=b"Input voltage is 12 V to 48 V DC with 34 W peak electrical power.",
    )
    client_history = [
        {"role": t["role"], "content": t["text"]} for t in loaded["turns"]
    ]
    list(
        chatbot_service.stream_chat(
            pid,
            message="What is the 12 V to 48 V DC power?",
            history=client_history,
        )
    )
    sent_messages = mock_backend.chat_calls[-1]["messages"]
    assert {"role": "user", "content": "What is the DC voltage range?"} in sent_messages
    assert {"role": "assistant", "content": "Peak power draw is 34 W [1]."} in sent_messages

    chatbot_service.delete_saved_chat(pid, cid)
    assert chatbot_service.list_saved_chats(pid) == []
    with pytest.raises(ChatNotFoundError):
        chatbot_service.get_saved_chat(pid, cid)


def test_reupload_same_filename_preserves_original_when_embed_fails(
    chatbot_service: ChatbotService,
    mock_backend: DeterministicMockBackend,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If backend.embed fails during a same-filename re-upload, the original doc and chunks survive."""
    prof = chatbot_service.create_profile({"name": "Safe Reupload Bot"})
    pid = prof["id"]

    v1_bytes = b"# Warranty\n\nThe standard hardware warranty is 36-month return-to-depot coverage."
    res_v1 = chatbot_service.ingest_document(pid, filename="warranty.md", data=v1_bytes)
    orig_doc = res_v1["document"]
    assert chatbot_service.vectorstore.inspect_collection(pid).chunk_count == 1

    real_embed = mock_backend.embed

    def failing_embed(texts: list[str], model: str) -> list[list[float]]:
        raise ModelLoadError(model, details="Simulated embedding failure during re-upload")

    monkeypatch.setattr(mock_backend, "embed", failing_embed)

    v2_bytes = b"# Warranty\n\nUpdated policy: 60-month AstraCare extended coverage."
    with pytest.raises(ModelLoadError):
        chatbot_service.ingest_document(pid, filename="warranty.md", data=v2_bytes)

    # Original document record, stored file, and Chroma chunks must remain intact
    docs_after_fail = chatbot_service.list_documents(pid)
    assert len(docs_after_fail) == 1
    assert docs_after_fail[0]["id"] == orig_doc["id"]
    assert docs_after_fail[0]["file_hash"] == orig_doc["file_hash"]
    assert (chatbot_service.profiles.docs_dir(pid) / orig_doc["stored_filename"]).exists()
    assert chatbot_service.vectorstore.inspect_collection(pid).chunk_count == 1

    # Restore embed and verify original chunks are still queryable, and a valid re-upload replaces v1
    monkeypatch.setattr(mock_backend, "embed", real_embed)
    profile_obj = chatbot_service.profiles.get_profile(pid)
    ans = chatbot_service.rag.query(profile=profile_obj, question="What is the 36-month warranty?")
    assert ans.refused is False
    assert "36-month" in ans.answer

    res_v2 = chatbot_service.ingest_document(pid, filename="warranty.md", data=v2_bytes)
    docs_after_ok = chatbot_service.list_documents(pid)
    assert len(docs_after_ok) == 1
    assert docs_after_ok[0]["id"] == res_v2["document"]["id"]
    assert docs_after_ok[0]["id"] != orig_doc["id"]
    assert chatbot_service.vectorstore.inspect_collection(pid).chunk_count == 1


def test_atomic_json_writes_and_corrupt_profile_warning_log(
    chatbot_service: ChatbotService,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Failed JSON writes preserve existing file without leftover temp files; corrupt profiles log a warning."""
    target = tmp_path / "sample.json"
    atomic_write_json(target, {"status": "original"})
    assert '"original"' in target.read_text(encoding="utf-8")

    with pytest.raises(TypeError):
        atomic_write_json(target, {"bad": object()})

    assert '"original"' in target.read_text(encoding="utf-8")
    assert list(tmp_path.glob("*.tmp")) == []

    good = chatbot_service.create_profile({"name": "Healthy Profile"})
    corrupt_dir = chatbot_service.profiles.profiles_dir / "corrupt-bot"
    corrupt_dir.mkdir(parents=True, exist_ok=True)
    corrupt_file = corrupt_dir / "profile.json"
    corrupt_file.write_text("{truncated json", encoding="utf-8")

    with caplog.at_level(logging.WARNING, logger="app.core.profiles"):
        listed = chatbot_service.list_profiles()

    assert [p["id"] for p in listed] == [good["id"]]
    assert any(str(corrupt_file) in rec.getMessage() for rec in caplog.records)


def test_describe_profile_changes_pinned_strings_and_with_updates() -> None:
    """Pin exact describe_profile_changes output strings and Profile.with_updates coercion/immutability."""
    before = Profile(
        id="bot-1",
        name="Alpha Bot",
        preamble="Old preamble",
        chat_model="qwen3.5:4b",
        embedding_model="nomic-embed-text",
        temperature=0.20,
        top_k=5,
        similarity_threshold=0.50,
        chunk_size=1000,
        chunk_overlap=150,
        history_turns=4,
        enable_thinking=False,
        settings_history=[{"saved_at": "2026-01-01T00:00:00Z", "changes": ["Initial"]}],
        created_at="2026-01-01T00:00:00Z",
        updated_at="2026-01-01T00:00:00Z",
    )

    after = before.with_updates(
        {
            "id": "tampered-id",
            "created_at": "2099-01-01T00:00:00Z",
            "updated_at": "2099-01-01T00:00:00Z",
            "settings_history": [],
            "unknown_field": "ignored",
            "name": "  Beta Bot  ",
            "preamble": "New preamble",
            "chat_model": " llama3.2:3b ",
            "embedding_model": " mxbai-embed-large ",
            "similarity_threshold": "0.65",
            "temperature": "0.45",
            "top_k": "8",
            "chunk_size": "1200",
            "chunk_overlap": "200",
            "history_turns": "6",
            "refusal_messages": {"en": " New refusal "},
            "enable_thinking": 1,
        }
    )

    # Immutable keys must not be altered by with_updates
    assert after.id == "bot-1"
    assert after.created_at == "2026-01-01T00:00:00Z"
    assert after.updated_at == "2026-01-01T00:00:00Z"
    assert len(after.settings_history) == 1

    assert describe_profile_changes(before, after) == [
        "Name (Alpha Bot → Beta Bot)",
        "System Preamble",
        "Chat Model (qwen3.5:4b → llama3.2:3b)",
        "Embedding Model (nomic-embed-text → mxbai-embed-large)",
        "Threshold (0.50 → 0.65)",
        "Temperature (0.20 → 0.45)",
        "Top-K (5 → 8)",
        "Chunk Size (1000 → 1200)",
        "Chunk Overlap (150 → 200)",
        "History Turns (4 → 6)",
        "Refusal messages",
        "Thinking Mode (On)",
    ]

    toggled_off = after.with_updates({"enable_thinking": False, "temperature": 0.4502})
    assert describe_profile_changes(after, toggled_off) == ["Thinking Mode (Off)"]


def test_editing_only_non_english_refusal_message_records_change(
    chatbot_service: ChatbotService,
):
    """1g: Editing only a non-English refusal message records 'Refusal messages' in history."""
    prof = chatbot_service.create_profile({"name": "Refusal Bot"})
    pid = prof["id"]
    ms_refusal_new = "Maaf, maklumat tiada dalam pangkalan pengetahuan."
    updated_prof = chatbot_service.update_profile(
        pid,
        {"refusal_messages": {"ms": ms_refusal_new}},
    )
    history = updated_prof["settings_history"]
    assert len(history) >= 1
    latest_entry = history[-1]
    assert "Refusal messages" in latest_entry["changes"]
