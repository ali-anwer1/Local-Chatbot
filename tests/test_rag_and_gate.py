"""Unit tests for the code-enforced refusal gate, grounded citations, and multi-model profiles (§7.2, §7.3, §7.8)."""

from __future__ import annotations

from pathlib import Path

from app.core.rag import (
    FIXED_GROUNDING_RULES,
    build_retrieval_query,
    classify_conversational_intent,
    extract_cited_indices,
)
from app.core.service import ChatbotService
from samples.generate_samples import main as generate_samples
from tests.conftest import DeterministicMockBackend


def test_refusal_gate_blocks_out_of_scope_without_calling_llm(
    chatbot_service: ChatbotService,
    mock_backend: DeterministicMockBackend,
) -> None:
    """When best similarity < threshold, refuse with reason and NEVER invoke backend.chat (§7.3)."""
    prof = chatbot_service.create_profile(
        {
            "name": "Strict Gate Bot",
            "similarity_threshold": 0.50,
            "refusal_message": "I can't answer that from my knowledge base.",
        }
    )
    pid = prof["id"]
    chatbot_service.ingest_document(
        pid,
        filename="hardware.txt",
        data=b"The ARM Cortex-A78AE processor includes 8 GB ECC LPDDR5 RAM and 64 GB eMMC storage.",
    )

    assert len(mock_backend.chat_calls) == 0

    profile_obj = chatbot_service.profiles.get_profile(pid)
    out_of_scope = chatbot_service.rag.query(
        profile=profile_obj,
        question="How do I bake a chocolate soufflé in a convection oven?",
    )

    assert out_of_scope.refused is True
    assert out_of_scope.answer == "I can't answer that from my knowledge base."
    assert "No relevant content found in this bot's knowledge base." in out_of_scope.refusal_reason
    assert "threshold is 0.50" in out_of_scope.refusal_reason
    assert out_of_scope.sources == []
    # CRITICAL (§7.3): LLM chat() must NEVER be called when the gate fails!
    assert len(mock_backend.chat_calls) == 0


def test_mixed_format_documents_return_accurate_file_and_page_citations(
    chatbot_service: ChatbotService,
    mock_backend: DeterministicMockBackend,
) -> None:
    """Ingest 4 mixed-format files (PDF, DOCX, MD, TXT) and verify grounded citations (§7.1, §7.2)."""
    generate_samples()
    samples_dir = Path(__file__).resolve().parent.parent / "samples" / "docs"

    prof = chatbot_service.create_profile(
        {
            "name": "Astra Demo Profile",
            "preamble": "You are the Astra NX-200 hardware assistant.",
            "similarity_threshold": 0.40,
        }
    )
    pid = prof["id"]

    for fname in (
        "astra_nx200_datasheet.pdf",
        "astra_nx200_maintenance_sop.docx",
        "astra_warranty_and_sla.md",
        "astra_field_faq.txt",
    ):
        chatbot_service.ingest_document(
            pid,
            filename=fname,
            data=(samples_dir / fname).read_bytes(),
        )

    profile_obj = chatbot_service.profiles.get_profile(pid)

    # Query PDF Page 2
    ans_pdf = chatbot_service.rag.query(
        profile=profile_obj,
        question="What is the DC input voltage range and 34 W peak electrical power?",
    )
    assert ans_pdf.refused is False
    assert any(
        s["source_file"] == "astra_nx200_datasheet.pdf" and s["page"] == 2
        for s in ans_pdf.sources
    )
    # Verify fixed grounding rules were appended to system prompt
    system_msg = mock_backend.chat_calls[-1]["messages"][0]["content"]
    assert FIXED_GROUNDING_RULES in system_msg

    # Query DOCX Section
    ans_docx = chatbot_service.rag.query(
        profile=profile_obj,
        question="What thermal pad AST-TPAD-15 and 0.65 M3 torque are required?",
    )
    assert ans_docx.refused is False
    assert ans_docx.sources[0]["source_file"] == "astra_nx200_maintenance_sop.docx"
    assert ans_docx.sources[0]["section"] == "Thermal Interface Pad Replacement"


def test_two_profiles_with_different_chat_models_both_work(
    chatbot_service: ChatbotService,
    mock_backend: DeterministicMockBackend,
) -> None:
    """Two profiles configured with different installed chat models route to each model (§7.8)."""
    p1 = chatbot_service.create_profile({"name": "Qwen Bot", "chat_model": "qwen3.5:4b"})
    p2 = chatbot_service.create_profile({"name": "Llama Bot", "chat_model": "llama3.2:3b"})

    doc_bytes = b"The standard hardware warranty is 36-month coverage, extendable to 60-month AstraCare."
    chatbot_service.ingest_document(p1["id"], filename="w.txt", data=doc_bytes)
    chatbot_service.ingest_document(p2["id"], filename="w.txt", data=doc_bytes)

    r1 = chatbot_service.rag.query(
        profile=chatbot_service.profiles.get_profile(p1["id"]),
        question="How long is the 36-month warranty?",
    )
    r2 = chatbot_service.rag.query(
        profile=chatbot_service.profiles.get_profile(p2["id"]),
        question="How long is the 36-month warranty?",
    )

    assert r1.refused is False and "qwen3.5:4b" in r1.answer
    assert r2.refused is False and "llama3.2:3b" in r2.answer
    assert mock_backend.chat_calls[-2]["model"] == "qwen3.5:4b"
    assert mock_backend.chat_calls[-1]["model"] == "llama3.2:3b"


def test_followup_pronoun_query_appends_previous_user_message() -> None:
    history = [
        {"role": "user", "content": "Tell me about the Astra NX-200 gateway."},
        {"role": "assistant", "content": "It is a fanless industrial edge gateway [1]."},
    ]
    expanded = build_retrieval_query("What about its warranty?", history)
    assert "Astra NX-200 gateway" in expanded
    assert "What about its warranty?" in expanded

    standalone = build_retrieval_query(
        "What galvanic isolation rating do the RS-485 serial ports provide on page three?",
        history,
    )
    assert "Astra NX-200 gateway" not in standalone


def test_conversational_and_capability_queries_answered_naturally_while_outside_refused(
    chatbot_service: ChatbotService,
    mock_backend: DeterministicMockBackend,
) -> None:
    """Greetings and 'What can you help me with?' succeed naturally, while world trivia is refused."""
    prof = chatbot_service.create_profile(
        {
            "name": "Capability Bot",
            "preamble": "You assist engineers with Astra NX-200 hardware specifications.",
            "similarity_threshold": 0.55,
        }
    )
    pid = prof["id"]
    chatbot_service.ingest_document(
        pid,
        filename="nx200_specs.md",
        data=b"# Hardware\n\nThe ARM Cortex-A78AE processor includes 8 GB ECC LPDDR5 RAM.",
    )
    docs_overview = chatbot_service.list_documents(pid)
    profile_obj = chatbot_service.profiles.get_profile(pid)

    assert classify_conversational_intent("What can you help me with?") == "capability"
    assert classify_conversational_intent("Hi there!") == "greeting"
    assert classify_conversational_intent("Thanks so much!") == "greeting"
    assert classify_conversational_intent("Hi, what is the capital of France?") is None

    cap_resp = chatbot_service.rag.query(
        profile=profile_obj,
        question="What can you help me with?",
        documents_overview=docs_overview,
    )
    assert cap_resp.refused is False
    assert cap_resp.conversational is True
    assert cap_resp.sources == []
    # Verify uploaded document list was injected into the capability prompt
    sent_user_msg = mock_backend.chat_calls[-1]["messages"][-1]["content"]
    assert "nx200_specs.md" in sent_user_msg

    # Pure greetings must not pollute follow-up retrieval query expansion
    greeting_history = [
        {"role": "user", "content": "Hi!"},
        {"role": "assistant", "content": "Hello! How can I help?"},
    ]
    assert build_retrieval_query("What RAM is used?", greeting_history) == "What RAM is used?"

    # Outside world-knowledge questions blended with a greeting must STILL hit the refusal gate
    calls_before = len(mock_backend.chat_calls)
    trivia = chatbot_service.rag.query(
        profile=profile_obj,
        question="Hi, what is the capital of France?",
        documents_overview=docs_overview,
    )
    assert trivia.refused is True
    assert len(mock_backend.chat_calls) == calls_before


def test_sources_filtered_strictly_to_cited_passages(
    chatbot_service: ChatbotService,
    mock_backend: DeterministicMockBackend,
) -> None:
    """Only passages [n] actually cited in the answer appear in sources; debug_chunks retains all top_k."""
    assert extract_cited_indices("See [2] and [1, 3] plus [Source 4].", max_index=4) == [2, 1, 3, 4]
    assert extract_cited_indices("Passages [2-3] confirm this, ignoring [99].", max_index=4) == [2, 3]
    assert extract_cited_indices("Uncited statement.", max_index=4) == []

    prof = chatbot_service.create_profile(
        {
            "name": "Citation Filter Bot",
            "top_k": 3,
            "similarity_threshold": 0.20,
        }
    )
    pid = prof["id"]
    chatbot_service.ingest_document(
        pid,
        filename="power_a.txt",
        data=b"Primary input voltage is 12 V DC electrical power.",
    )
    chatbot_service.ingest_document(
        pid,
        filename="power_b.txt",
        data=b"Maximum input voltage is 48 V DC with 34 W peak power.",
    )
    chatbot_service.ingest_document(
        pid,
        filename="power_c.txt",
        data=b"Electrical power temperature derating starts above 65 C.",
    )

    # Force mock LLM to cite ONLY passage [2] out of the 3 retrieved chunks
    mock_backend.next_chat_reply = "The peak power rating is 34 W at up to 48 V DC [2]."
    resp = chatbot_service.rag.query(
        profile=chatbot_service.profiles.get_profile(pid),
        question="What is the 48 V DC voltage and 34 W peak electrical power?",
    )

    assert resp.refused is False
    assert len(resp.debug_chunks) == 3
    assert len(resp.sources) == 1
    assert resp.sources[0]["index"] == 2
    cited_flags = {c["index"]: c["cited"] for c in resp.debug_chunks}
    assert cited_flags == {1: False, 2: True, 3: False}


def test_live_status_events_and_thinking_mode_default_off(
    chatbot_service: ChatbotService,
    mock_backend: DeterministicMockBackend,
) -> None:
    """Streaming query emits live 'status' progress events and disables thinking by default."""
    prof = chatbot_service.create_profile({"name": "Fast Mode Bot"})
    pid = prof["id"]
    assert prof["enable_thinking"] is False

    chatbot_service.ingest_document(
        pid,
        filename="warranty.txt",
        data=b"Standard warranty is 36-month hardware coverage.",
    )
    events = list(
        chatbot_service.stream_chat(
            pid,
            message="What is the 36-month warranty?",
        )
    )
    event_types = [e["event"] for e in events]
    assert "status" in event_types
    assert event_types.index("status") < event_types.index("token")
    assert mock_backend.chat_calls[-1]["options"]["think"] is False

    # User explicitly toggles thinking on for a query
    list(
        chatbot_service.stream_chat(
            pid,
            message="What is the 36-month warranty?",
            runtime_options={"think": True},
        )
    )
    assert mock_backend.chat_calls[-1]["options"]["think"] is True


def test_query_returns_real_language_rule_and_confidence(
    chatbot_service: ChatbotService,
):
    """1a: RAGEngine.query() returns real rule ('selector') and confidence when requested_language is set."""
    prof = chatbot_service.create_profile({"name": "Lang Rule Bot"})
    pid = prof["id"]
    chatbot_service.ingest_document(
        pid,
        filename="notes.txt",
        data=b"Industrial gateway operates at 24V DC input voltage.",
    )
    profile_obj = chatbot_service.profiles.get_profile(pid)
    resp = chatbot_service.rag.query(
        profile=profile_obj,
        question="What is the input voltage?",
        requested_language="ms",
    )
    assert resp.target_language == "ms"
    assert resp.language_rule == "selector"
    assert resp.language_confidence > 0.0


def test_build_retrieval_query_does_not_prefix_full_length_chinese_question():
    """1e: build_retrieval_query does not treat full-length Chinese question as follow-up."""
    from app.core.rag import build_retrieval_query

    history = [
        {"role": "user", "content": "Tell me about the power specifications."},
        {"role": "assistant", "content": "The system requires 24V DC."},
    ]
    full_cjk_q = "工业网关支持哪些无线通信协议和接口？"
    query = build_retrieval_query(full_cjk_q, history=history)
    # Must NOT prepend the previous user message
    assert query == full_cjk_q
    assert "power specifications" not in query

    # Short CJK question (< 10 chars) should still be treated as follow-up
    short_cjk_q = "接口呢"
    query_short = build_retrieval_query(short_cjk_q, history=history)
    assert query_short == f"Tell me about the power specifications. {short_cjk_q}"

    cjk_warranty = "保修期呢"
    query_cjk_warranty = build_retrieval_query(cjk_warranty, history=history)
    assert query_cjk_warranty == f"Tell me about the power specifications. {cjk_warranty}"

    # English short follow-up (<= 6 words or < 40 chars)
    en_warranty = "What about the warranty period?"
    query_en = build_retrieval_query(en_warranty, history=history)
    assert query_en == f"Tell me about the power specifications. {en_warranty}"
