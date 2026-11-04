"""Comprehensive unit tests for multilingual answers, language resolution, refusals, and patterns (§ Part 1)."""

from __future__ import annotations

from typing import Any

from app.core.languages import (
    SUPPORTED_LANGUAGES,
    SUPPORTED_LANGUAGE_CODES,
    LanguageResolution,
    detect_explicit_language_request,
    detect_language,
    resolve_target_language,
)
from app.core.profiles import Profile
from app.core.rag import (
    CAPABILITY_QUERY_PATTERN,
    FOLLOWUP_PRONOUN_PATTERN,
    PURE_GREETING_PATTERN,
    assemble_conversational_messages,
    assemble_rag_messages,
    classify_conversational_intent,
)
from app.core.service import ChatbotService
from tests.conftest import DeterministicMockBackend


def test_supported_languages_definition():
    """Verify 8 supported languages are defined with required attributes."""
    expected_codes = {"en", "ms", "zh", "ta", "id", "ja", "ko", "ar"}
    assert set(SUPPORTED_LANGUAGE_CODES) == expected_codes
    assert len(SUPPORTED_LANGUAGES) == 8

    for lang in SUPPORTED_LANGUAGES:
        assert lang.code in expected_codes
        assert lang.english_name
        assert lang.native_name
        assert lang.instruction_name
        assert lang.default_refusal_message


def test_language_detection_and_override_precedence():
    """Verify strict precedence: selector > explicit request > detected > fallback."""
    # 1. Selector overrides explicit request, detected language, and fallback
    res1 = resolve_target_language(
        latest_user_message="Tolong jawab dalam bahasa Melayu.",
        requested_language="ja",
        history=None,
        profile_default_language="en",
    )
    assert res1.target_language == "ja"
    assert res1.rule == "selector"

    # 2. Explicit request in question overrides detected language and fallback
    res2 = resolve_target_language(
        latest_user_message="Apakah spesifikasi bateri NX-200? Please answer in English.",
        requested_language="auto",
        history=None,
        profile_default_language="ms",
    )
    assert res2.target_language == "en"
    assert res2.rule == "explicit request"

    # Malay explicit request
    res2b = resolve_target_language(
        latest_user_message="What is the warranty period? Jawab dalam bahasa Melayu.",
        requested_language="auto",
        history=None,
        profile_default_language="en",
    )
    assert res2b.target_language == "ms"
    assert res2b.rule == "explicit request"

    # Chinese explicit request
    res2c = resolve_target_language(
        latest_user_message="What is the warranty period? 请用中文回答",
        requested_language="auto",
        history=None,
        profile_default_language="en",
    )
    assert res2c.target_language == "zh"
    assert res2c.rule == "explicit request"

    # 3. Detected language when confidence is sufficient and >= 3 words
    res3_zh = resolve_target_language(
        latest_user_message="如何重置工业网关？请说明步骤。",
        requested_language="auto",
        history=None,
        profile_default_language="en",
    )
    assert res3_zh.target_language == "zh"
    assert res3_zh.rule == "detected"

    # 4. Fallback on short message (< 3 words) without prior history turns to default_language
    res4 = resolve_target_language(
        latest_user_message="ok",
        requested_language="auto",
        history=None,
        profile_default_language="en",
    )
    assert res4.target_language == "en"
    assert res4.rule == "fallback"

    # 5. Fallback on short message uses previous user turn if available
    history = [
        {"role": "user", "content": "Bagaimanakah cara untuk menyelenggara peranti ini?"},
        {"role": "assistant", "content": "Berikut adalah panduan penyelenggaraan..."},
    ]
    res5 = resolve_target_language(
        latest_user_message="ok faham",
        requested_language="auto",
        history=history,
        profile_default_language="en",
    )
    assert res5.target_language == "ms"
    assert res5.rule == "fallback"


def test_malay_indonesian_tie_break():
    """Verify tie-break prefers Malay unless Indonesian has a significantly higher confidence margin."""
    # When text is detected as Indonesian but Malay is close, prefer Malay
    # 'Bagaimana cara membuka pintu ini' is valid in both, Malaysian context tie-break should prefer Malay (ms)
    code, conf = detect_language("Bagaimana cara membuka pintu ini dengan selamat dan betul?")
    assert code == "ms"

    # Pure Indonesian specific vocabulary
    code_id, _ = detect_language("Bisa tolong jelaskan gimana cara aktivasi perangkat di Jakarta?")
    assert code_id in ("id", "ms")


def test_greeting_and_capability_patterns():
    """Verify regex patterns match Malay and Chinese greetings and capability queries."""
    # Malay greetings
    assert PURE_GREETING_PATTERN.match("hai")
    assert PURE_GREETING_PATTERN.match("Hai!")
    assert PURE_GREETING_PATTERN.match("helo")
    assert PURE_GREETING_PATTERN.match("selamat pagi")
    assert PURE_GREETING_PATTERN.match("terima kasih")
    assert PURE_GREETING_PATTERN.match("terima kasih banyak")

    # Chinese greetings
    assert PURE_GREETING_PATTERN.match("你好")
    assert PURE_GREETING_PATTERN.match("您好")
    assert PURE_GREETING_PATTERN.match("早上好")
    assert PURE_GREETING_PATTERN.match("谢谢")
    assert PURE_GREETING_PATTERN.match("多谢")

    # Capability queries
    assert CAPABILITY_QUERY_PATTERN.search("apa yang boleh awak bantu")
    assert CAPABILITY_QUERY_PATTERN.search("apa yang kamu boleh buat?")
    assert CAPABILITY_QUERY_PATTERN.search("你能帮我什么")
    assert CAPABILITY_QUERY_PATTERN.search("你能做什么")

    # Follow-up pronouns
    assert FOLLOWUP_PRONOUN_PATTERN.search("Bagaimana dengan harganya?")
    assert FOLLOWUP_PRONOUN_PATTERN.search("Adakah ia kalis air?")
    assert FOLLOWUP_PRONOUN_PATTERN.search("它支持多大功率？")
    assert FOLLOWUP_PRONOUN_PATTERN.search("这个设备的重量是多少？")


def test_user_payload_ends_with_language_instruction():
    """Verify that assemble_rag_messages and assemble_conversational_messages append 'Answer in <instruction name>.' as the final line."""
    from app.core.vectorstore import RetrievedChunk

    profile = Profile(id="test-bot", name="Test Bot")

    chunk = RetrievedChunk(
        chunk_id="chunk-1",
        text="Kapasiti bateri ialah 5000mAh.",
        score=0.9,
        distance=0.1,
        source_file="spec.txt",
        chunk_index=0,
        file_hash="hash1",
    )

    # RAG messages
    rag_msgs = assemble_rag_messages(
        profile=profile,
        question="Berapakah kapasiti bateri NX-200?",
        retrieved_chunks=[chunk],
        history=[],
        target_language="ms",
    )
    last_user_rag = rag_msgs[-1]["content"]
    assert last_user_rag.endswith("Answer in Bahasa Melayu (Malay).")
    # Grounding instructions include language constraint
    assert "Answer in Bahasa Melayu (Malay)." in rag_msgs[0]["content"]

    # Conversational messages
    conv_msgs = assemble_conversational_messages(
        profile=profile,
        question="Hai selamat pagi!",
        documents_overview=[],
        sample_snippets=[],
        history=[],
        target_language="zh",
    )
    last_user_conv = conv_msgs[-1]["content"]
    assert last_user_conv.endswith("Answer in Simplified Chinese.")
    assert "Answer in Simplified Chinese." in conv_msgs[0]["content"]


def test_profile_migration_preserves_refusal_and_adds_defaults():
    """Legacy profile dictionary with single refusal_message migrates smoothly to refusal_messages dict."""
    legacy_data: dict[str, Any] = {
        "id": "legacy-prof",
        "name": "Legacy Bot",
        "refusal_message": "Custom English refusal.",
    }
    profile = Profile.from_dict(legacy_data)
    assert profile.refusal_message == "Custom English refusal."
    assert profile.refusal_messages["en"] == "Custom English refusal."
    # Other languages are populated with built-in defaults
    assert profile.refusal_messages["ms"] == "Saya tidak dapat menjawab soalan itu berdasarkan pangkalan pengetahuan saya."
    assert profile.refusal_messages["zh"] == "我无法根据知识库回答该问题。"
    assert profile.default_language == "en"

    # get_refusal_message method
    assert profile.get_refusal_message("en") == "Custom English refusal."
    assert profile.get_refusal_message("ms") == "Saya tidak dapat menjawab soalan itu berdasarkan pangkalan pengetahuan saya."


def test_refusal_gate_uses_target_language_on_code_path(
    chatbot_service: ChatbotService,
    mock_backend: DeterministicMockBackend,
):
    """When code gate refuses an out-of-scope question, it uses the target language's refusal string."""
    prof = chatbot_service.create_profile(
        {
            "name": "Multilingual Strict Gate Bot",
            "similarity_threshold": 0.60,
            "refusal_message": "Cannot answer from KB.",
            "refusal_messages": {
                "en": "Cannot answer from KB.",
                "ms": "Pangkalan pengetahuan tidak mengandungi jawapan.",
                "zh": "知识库不包含此答案。",
            },
        }
    )
    pid = prof["id"]
    chatbot_service.ingest_document(
        pid,
        filename="hardware.txt",
        data=b"The device operates on 24V DC power with 10A current limit.",
    )

    profile_obj = chatbot_service.profiles.get_profile(pid)

    # 1. Out-of-scope question in Malay
    resp_ms = chatbot_service.rag.query(
        profile=profile_obj,
        question="Bagaimanakah cara memasak nasi lemak kukus yang sedap?",
        requested_language="auto",
    )
    assert resp_ms.refused is True
    assert "Pangkalan pengetahuan tidak mengandungi jawapan." in resp_ms.answer
    assert resp_ms.target_language == "ms"
    assert len(mock_backend.chat_calls) == 0

    # 2. Out-of-scope question in Chinese
    resp_zh = chatbot_service.rag.query(
        profile=profile_obj,
        question="如何做巧克力舒芙蕾甜点？",
        requested_language="auto",
    )
    assert resp_zh.refused is True
    assert "知识库不包含此答案。" in resp_zh.answer
    assert resp_zh.target_language == "zh"
    assert len(mock_backend.chat_calls) == 0


def test_model_refusal_check_matches_non_english_refusal(
    chatbot_service: ChatbotService,
    mock_backend: DeterministicMockBackend,
):
    """If the LLM outputs a target-language refusal string, it is properly recognized as refused."""
    prof = chatbot_service.create_profile(
        {
            "name": "Model Refusal Bot",
            "similarity_threshold": 0.10,  # Low threshold so code gate passes
        }
    )
    pid = prof["id"]
    chatbot_service.ingest_document(
        pid,
        filename="doc.txt",
        data=b"Industrial NX-200 gateway technical overview and operating instructions.",
    )

    profile_obj = chatbot_service.profiles.get_profile(pid)
    ms_refusal = profile_obj.get_refusal_message("ms")

    # Queue mock backend response that starts with the Malay refusal text
    mock_backend.next_chat_reply = f"{ms_refusal} Dokumen tidak menyebut tentang perincian tersebut."

    resp = chatbot_service.rag.query(
        profile=profile_obj,
        question="Apakah protokol komunikasi tanpa wayar yang disokong?",
        requested_language="ms",
    )

    assert resp.refused is True
    assert resp.target_language == "ms"
    assert ms_refusal in resp.answer


def test_explicit_language_request_patterns_positive_and_negative():
    """1d: Explicit language request regexes require answer/reply verbs and reject mentions."""
    # Positive: explicit command to answer/reply in that language
    assert detect_explicit_language_request("Please answer in English.") == "en"
    assert detect_explicit_language_request("Write in English please.") == "en"
    assert detect_explicit_language_request("Tolong jawab dalam bahasa Melayu.") == "ms"
    assert detect_explicit_language_request("Sila balas dalam bahasa Malaysia.") == "ms"
    assert detect_explicit_language_request("请用中文回答我") == "zh"
    assert detect_explicit_language_request("请用简体中文回复") == "zh"
    assert detect_explicit_language_request("Tolong jawab dalam bahasa Indonesia.") == "id"
    assert detect_explicit_language_request("日本語で答えてください") == "ja"
    assert detect_explicit_language_request("한국어로 대답해주세요") == "ko"
    assert detect_explicit_language_request("أجب باللغة العربية") == "ar"

    # Negative: mentions of languages without answer/reply verbs must NOT force the language
    assert detect_explicit_language_request("Is the manual available in 中文?") is None
    assert detect_explicit_language_request("Adakah manual ini ada dalam bahasa Melayu?") is None
    assert detect_explicit_language_request("Apakah dokumen tersedia dalam bahasa Indonesia?") is None
    assert detect_explicit_language_request("Do you have documentation in 日本語?") is None
    assert detect_explicit_language_request("Can you use english as a search query?") is None
    assert detect_explicit_language_request("Does this system support Korean or 한국어?") is None
