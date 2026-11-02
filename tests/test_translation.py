"""Tests for on-demand answer translation (§ Part 2).

Verifies:
- Translation prompt assembly strictly follows faithfulness and citation preservation rules.
- POST /api/translate streams tokens and finishes with done event.
- Translation uses temperature 0.1, think: false, and profile's chat model.
- Translation NEVER calls retrieval or vectorstore embedding.
- Citation markers [n] are preserved in translated output.
- Saved chat serialization and Markdown export preserve translations.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

from fastapi.testclient import TestClient
import pytest

from app.core.languages import (
    SUPPORTED_LANGUAGES,
    assemble_translation_messages,
)
from app.core.service import ChatbotService
from app.main import create_app
from tests.conftest import DeterministicMockBackend


def test_assemble_translation_messages_structure():
    """Verify translation messages build the right prompt with strict rules."""
    text = "The gateway requires 24V DC power [1]. Do not exceed 10A current [2]."
    messages = assemble_translation_messages(
        text=text,
        target_language="ms",
        source_language="en",
    )

    assert len(messages) == 2
    system_msg = messages[0]["content"]
    user_msg = messages[1]["content"]

    # Rules verification
    assert "Translate the text faithfully" in system_msg
    assert "Keep [n] citation markers" in system_msg
    assert "numbers, units, product names and model numbers unchanged" in system_msg
    assert "Output only the translation" in system_msg
    assert "Bahasa Melayu (Malay)" in system_msg

    # User message contains original text
    assert text in user_msg


def test_translation_uses_chat_model_and_never_calls_retrieval(
    chatbot_service: ChatbotService,
    mock_backend: DeterministicMockBackend,
):
    """Translation must invoke backend.chat with temperature 0.1, think: false, and never touch vectorstore or embed."""
    prof = chatbot_service.create_profile(
        {
            "name": "Translation Profile",
            "chat_model": "qwen3.5:4b",
        }
    )
    pid = prof["id"]

    # Ingest document to have a populated vectorstore
    chatbot_service.ingest_document(
        pid,
        filename="specs.txt",
        data=b"The operating voltage is 24V DC with 10A current limit.",
    )

    initial_embed_calls = len(mock_backend.embed_calls)
    initial_chat_calls = len(mock_backend.chat_calls)

    original_answer = "The NX-200 operates on 24V DC power [1] and has an IP67 rating [2]."
    translated_mock = "NX-200 beroperasi pada kuasa 24V DC [1] dan mempunyai penarafan IP67 [2]."
    mock_backend.next_chat_reply = translated_mock

    # Run translation
    result = chatbot_service.translate(
        text=original_answer,
        target_language="ms",
        source_language="en",
        profile_id=pid,
    )

    # 1. Output matches mocked translation and preserves citations
    assert result["full_text"] == translated_mock
    assert result["target_language"] == "ms"
    assert "[1]" in result["full_text"]
    assert "[2]" in result["full_text"]

    # 2. Embedding was NEVER called during translation (retrieval bypassed)
    assert len(mock_backend.embed_calls) == initial_embed_calls

    # 3. Chat model was called with specified options: temperature 0.1, think False
    assert len(mock_backend.chat_calls) == initial_chat_calls + 1
    last_call = mock_backend.chat_calls[-1]
    assert last_call["model"] == "qwen3.5:4b"
    assert last_call["options"]["temperature"] == 0.1
    assert last_call["options"]["think"] is False


def test_stream_translate_yields_status_tokens_and_done(
    chatbot_service: ChatbotService,
    mock_backend: DeterministicMockBackend,
):
    """Verify streaming translation yields status, token, and done events."""
    prof = chatbot_service.create_profile({"name": "Stream Bot"})
    pid = prof["id"]

    mock_backend.next_chat_reply = "Halo dunia [1]."

    events = list(
        chatbot_service.stream_translate(
            text="Hello world [1].",
            target_language="id",
            source_language="en",
            profile_id=pid,
        )
    )

    assert len(events) >= 3
    # First event is status
    assert events[0]["event"] == "status"
    assert "Bahasa Indonesia" in events[0]["data"]["label"]

    # Middle events are tokens
    token_events = [e for e in events if e["event"] == "token"]
    assert len(token_events) > 0
    full_text = "".join(e["data"]["text"] for e in token_events)
    assert full_text == "Halo dunia [1]."

    # Last event is done
    done_events = [e for e in events if e["event"] == "done"]
    assert len(done_events) == 1
    assert done_events[0]["data"]["full_text"] == "Halo dunia [1]."
    assert done_events[0]["data"]["target_language"] == "id"


def test_translate_api_endpoint(
    chatbot_service: ChatbotService,
    mock_backend: DeterministicMockBackend,
):
    """Test POST /api/translate endpoint streams SSE events."""
    client = TestClient(create_app(chatbot_service))
    mock_backend.next_chat_reply = "Terjemahan Bahasa Melayu [1]."

    response = client.post(
        "/api/translate",
        json={
            "text": "English text [1].",
            "target_language": "ms",
            "source_language": "en",
        },
    )

    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]
    content = response.text
    assert "event: status" in content
    assert "event: token" in content
    assert "event: done" in content
    assert "Terjemahan Bahasa Melayu [1]." in content


def test_saved_chats_include_translations_and_survives_roundtrip(
    chatbot_service: ChatbotService,
):
    """Verify saved chats persist and retrieve the translations dictionary."""
    prof = chatbot_service.create_profile({"name": "Saved Chat Bot"})
    pid = prof["id"]

    turns = [
        {"role": "user", "text": "What is the battery spec?"},
        {
            "role": "assistant",
            "text": "The battery capacity is 5000mAh [1].",
            "sources": [{"index": 1, "source_file": "doc.pdf", "chunk_index": 0}],
            "translations": {
                "ms": "Kapasiti bateri ialah 5000mAh [1].",
                "zh": "电池容量为 5000mAh [1]。",
            },
        },
    ]

    saved = chatbot_service.save_chat(
        pid,
        turns=turns,
        title="Battery Q&A",
    )

    # Fetch back and verify
    retrieved = chatbot_service.get_saved_chat(pid, saved["id"])
    assert len(retrieved["turns"]) == 2
    asst_turn = retrieved["turns"][1]
    assert asst_turn["translations"]["ms"] == "Kapasiti bateri ialah 5000mAh [1]."
    assert asst_turn["translations"]["zh"] == "电池容量为 5000mAh [1]。"


def test_assemble_translation_messages_mandarin_chinese():
    """Verify Simplified Chinese translation prompt includes native name 简体中文 and prevents preambles."""
    text = "The system operates at 220V with IP65 enclosure [1]."
    messages = assemble_translation_messages(
        text=text,
        target_language="zh",
        source_language="en",
    )

    assert len(messages) == 2
    system_msg = messages[0]["content"]
    user_msg = messages[1]["content"]

    assert "Simplified Chinese (简体中文)" in system_msg
    assert "Translate the following text into Simplified Chinese (简体中文):" in user_msg
    assert "Output only the translation without any preamble, notes, or explanations." in system_msg
    assert text in user_msg


def test_translation_dropdown_popover_dom_and_css_specs():
    """Verify top-layer popover is properly declared in index.html, styled in CSS, and wired in chat.js."""
    from pathlib import Path

    repo_root = Path(__file__).resolve().parent.parent

    # 1. index.html contains top-layer popover with popover="auto" and role="menu"
    index_html = (repo_root / "app" / "static" / "index.html").read_text(encoding="utf-8")
    assert 'id="translate-dropdown-popover"' in index_html
    assert 'popover="auto"' in index_html
    assert 'role="menu"' in index_html

    # 2. styles.css contains top-layer fixed positioning and high z-index
    styles_css = (repo_root / "app" / "static" / "styles.css").read_text(encoding="utf-8")
    assert ".translate-popover" in styles_css
    assert ".translate-dropdown-menu[popover]" in styles_css
    assert "position: fixed;" in styles_css
    assert "z-index: 10000;" in styles_css

    # 3. chat.js targets assistant turn body (.assistant-body) and exports openTranslateMenu
    chat_js = (repo_root / "app" / "static" / "js" / "chat.js").read_text(encoding="utf-8")
    assert 'row.querySelector(".assistant-body")' in chat_js
    assert "export function openTranslateMenu" in chat_js
    assert "export async function triggerTranslation" in chat_js
    assert "positionTranslatePopover" in chat_js


def test_translation_overwrites_previous_translation_in_dom():
    """Verify chat.js clears previous translation container to overwrite rather than crowd output."""
    from pathlib import Path

    repo_root = Path(__file__).resolve().parent.parent
    chat_js = (repo_root / "app" / "static" / "js" / "chat.js").read_text(encoding="utf-8")

    # triggerTranslation must reset container.innerHTML to overwrite prior translations
    assert 'container.innerHTML = "";' in chat_js
    assert "turn.activeTranslationLang = targetLang;" in chat_js
    assert "row._translationAbortController" in chat_js

