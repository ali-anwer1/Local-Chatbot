"""Supported languages, detection, and explicit override rules for multilingual RAG."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import logging
import re
from typing import Any

from lingua import Language, LanguageDetectorBuilder

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SupportedLanguage:
    code: str
    english_name: str
    native_name: str
    instruction_name: str
    default_refusal_message: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


SUPPORTED_LANGUAGES: list[SupportedLanguage] = [
    SupportedLanguage(
        code="en",
        english_name="English",
        native_name="English",
        instruction_name="English",
        default_refusal_message="I can't answer that from my knowledge base.",
    ),
    SupportedLanguage(
        code="ms",
        english_name="Malay",
        native_name="Bahasa Melayu",
        instruction_name="Bahasa Melayu (Malay)",
        default_refusal_message="Saya tidak dapat menjawab soalan itu berdasarkan pangkalan pengetahuan saya.",
    ),
    SupportedLanguage(
        code="zh",
        english_name="Simplified Chinese",
        native_name="简体中文",
        instruction_name="Simplified Chinese",
        default_refusal_message="我无法根据知识库回答该问题。",
    ),
    SupportedLanguage(
        code="ta",
        english_name="Tamil",
        native_name="தமிழ்",
        instruction_name="Tamil",
        default_refusal_message="எனது அறிவுத் தளத்திலிருந்து இதற்குப் பதிலளிக்க முடியாது.",
    ),
    SupportedLanguage(
        code="id",
        english_name="Indonesian",
        native_name="Bahasa Indonesia",
        instruction_name="Bahasa Indonesia",
        default_refusal_message="Saya tidak dapat menjawab hal tersebut berdasarkan basis pengetahuan saya.",
    ),
    SupportedLanguage(
        code="ja",
        english_name="Japanese",
        native_name="日本語",
        instruction_name="Japanese",
        default_refusal_message="ナレッジベースからはその質問に答えることができません。",
    ),
    SupportedLanguage(
        code="ko",
        english_name="Korean",
        native_name="한국어",
        instruction_name="Korean",
        default_refusal_message="지식 기반에서는 해당 질문에 답할 수 없습니다.",
    ),
    SupportedLanguage(
        code="ar",
        english_name="Arabic",
        native_name="العربية",
        instruction_name="Arabic",
        default_refusal_message="لا يمكنني الإجابة عن ذلك من قاعدة المعرفة الخاصة بي.",
    ),
]

SUPPORTED_LANGUAGE_CODES: set[str] = {lang.code for lang in SUPPORTED_LANGUAGES}
LANGUAGE_BY_CODE: dict[str, SupportedLanguage] = {lang.code: lang for lang in SUPPORTED_LANGUAGES}
DEFAULT_REFUSAL_MESSAGES: dict[str, str] = {
    lang.code: lang.default_refusal_message for lang in SUPPORTED_LANGUAGES
}

# Lingua language mapping restricted strictly to supported languages
_LINGUA_MAP: dict[Language, str] = {
    Language.ENGLISH: "en",
    Language.MALAY: "ms",
    Language.CHINESE: "zh",
    Language.TAMIL: "ta",
    Language.INDONESIAN: "id",
    Language.JAPANESE: "ja",
    Language.KOREAN: "ko",
    Language.ARABIC: "ar",
}

# Create single detector instance once at module import
_DETECTOR = LanguageDetectorBuilder.from_languages(*_LINGUA_MAP.keys()).build()


# Explicit in-text language request patterns (precedence 3b)
# Every pattern strictly requires an answer/reply verb (§ Phase 1d)
EXPLICIT_LANGUAGE_REQUEST_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    # English
    (
        "en",
        re.compile(
            r"(?:"
            r"(?:answer|reply|respond|write)(?:\s+to\s+me)?\s+in\s+english|"
            r"(?:please\s+)?(?:answer|reply|respond|write)\s+in\s+english|"
            r"in\s+english\s*,\s*(?:please\s+)?(?:answer|reply|respond|write)"
            r")",
            re.IGNORECASE,
        ),
    ),
    # Malay
    (
        "ms",
        re.compile(
            r"(?:"
            r"(?:sila\s+|tolong\s+)?(?:jawab|balas|tulis)(?:\s+(?:kepada\s+saya|dalam))?\s+(?:bahasa\s+(?:melayu|malaysia)|melayu)|"
            r"(?:answer|reply|respond|write)\s+in\s+(?:malay|bahasa\s+melayu|bahasa\s+malaysia)"
            r")",
            re.IGNORECASE,
        ),
    ),
    # Simplified Chinese
    (
        "zh",
        re.compile(
            r"(?:"
            r"(?:请|请你)?(?:用|以)?(?:中文|汉语|普通话|简体中文)(?:来)?(?:回答|回复|作答)|"
            r"(?:请|请你)?(?:回答|回复|作答)(?:用|以)?(?:中文|汉语|普通话|简体中文)|"
            r"(?:please\s+)?(?:answer|reply|respond|write)\s+in\s+(?:chinese|mandarin|simplified\s+chinese)"
            r")",
            re.IGNORECASE,
        ),
    ),
    # Tamil
    (
        "ta",
        re.compile(
            r"(?:"
            r"(?:answer|reply|respond|write)\s+in\s+tamil|"
            r"தமிழில்\s+(?:பதிலளிக்கவும்|பதில்\s+அளிக்கவும்|பதில்\s+கூறவும்)"
            r")",
            re.IGNORECASE,
        ),
    ),
    # Indonesian
    (
        "id",
        re.compile(
            r"(?:"
            r"(?:tolong\s+|silakan\s+)?(?:jawab|balas|tulis)(?:\s+(?:kepada\s+saya|dalam))?\s+(?:bahasa\s+indonesia|indonesia)|"
            r"(?:answer|reply|respond|write)\s+in\s+indonesian"
            r")",
            re.IGNORECASE,
        ),
    ),
    # Japanese
    (
        "ja",
        re.compile(
            r"(?:"
            r"(?:answer|reply|respond|write)\s+in\s+japanese|"
            r"日本語で(?:答えて|返信して|回答して|お答えください)"
            r")",
            re.IGNORECASE,
        ),
    ),
    # Korean
    (
        "ko",
        re.compile(
            r"(?:"
            r"(?:answer|reply|respond|write)\s+in\s+korean|"
            r"한국어로\s+(?:대답해(?:주세요)?|답변해(?:주세요)?|답해(?:주세요)?|작성해(?:주세요)?)"
            r")",
            re.IGNORECASE,
        ),
    ),
    # Arabic
    (
        "ar",
        re.compile(
            r"(?:"
            r"(?:answer|reply|respond|write)\s+in\s+arabic|"
            r"(?:أجب|جاوب|رد)(?:\s+علي)?\s+(?:باللغة\s+العربية|بالعربية)"
            r")",
            re.IGNORECASE,
        ),
    ),
]


def detect_explicit_language_request(text: str) -> str | None:
    """Detect an explicit in-text request to answer in a particular language."""
    if not text:
        return None
    for code, pattern in EXPLICIT_LANGUAGE_REQUEST_PATTERNS:
        if pattern.search(text):
            return code
    return None


@dataclass(frozen=True)
class LanguageResolution:
    target_language: str
    instruction_name: str
    detected_language: str
    confidence: float
    rule: str  # "selector" | "explicit request" | "detected" | "fallback"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def detect_language(text: str) -> tuple[str, float]:
    """Detect language of text using the lingua detector restricted to supported languages.

    Applies the Malay/Indonesian tie-break rule: if 'id' is detected but confidence over 'ms'
    is not clearly higher (<= 0.20 margin), prefer 'ms' (Malaysian context).
    """
    cleaned = text.strip()
    if not cleaned:
        return "en", 0.0

    confidence_values = _DETECTOR.compute_language_confidence_values(cleaned)
    if not confidence_values:
        return "en", 0.0

    conf_by_code: dict[str, float] = {}
    for cv in confidence_values:
        code = _LINGUA_MAP.get(cv.language)
        if code:
            conf_by_code[code] = cv.value

    top_lang = confidence_values[0].language
    top_code = _LINGUA_MAP.get(top_lang, "en")
    top_conf = confidence_values[0].value

    # Tie-break rule: if id detected, check if margin over ms is clearly higher
    if top_code == "id":
        id_conf = conf_by_code.get("id", 0.0)
        ms_conf = conf_by_code.get("ms", 0.0)
        if (id_conf - ms_conf) <= 0.20:
            top_code = "ms"
            top_conf = max(id_conf, ms_conf)

    return top_code, round(top_conf, 3)


def is_short_text(text: str) -> bool:
    """Return True if text has under ~3 words, taking into account CJK scripts without spaces."""
    cleaned = text.strip()
    if not cleaned:
        return True
    cjk_chars = re.findall(r"[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]", cleaned)
    if cjk_chars:
        return len(cjk_chars) < 4
    words = [w for w in cleaned.split() if w]
    return len(words) < 3


def resolve_target_language(
    *,
    latest_user_message: str,
    history: list[dict[str, Any]] | None = None,
    profile_default_language: str = "en",
    requested_language: str | None = None,
) -> LanguageResolution:
    """Resolve the target response language following the prompt precedence:

    1. Per-chat selector in composer (if != 'auto' and valid) -> rule: 'selector'
    2. Explicit request in question text ('answer in English', '用中文回答') -> rule: 'explicit request'
    3. Detected language from latest user message (if text >= 3 words and confidence >= 0.35) -> rule: 'detected'
    4. Fallback: previous user message in chat, else profile default language -> rule: 'fallback'
    """
    cleaned = latest_user_message.strip()
    detected_code, confidence = detect_language(cleaned)

    # 1. Per-chat selector override (highest precedence)
    if requested_language and requested_language.strip().lower() != "auto":
        req_clean = requested_language.strip().lower()
        if req_clean in SUPPORTED_LANGUAGE_CODES:
            target_obj = LANGUAGE_BY_CODE[req_clean]
            return LanguageResolution(
                target_language=req_clean,
                instruction_name=target_obj.instruction_name,
                detected_language=detected_code,
                confidence=confidence,
                rule="selector",
            )

    # 2. Explicit request in question text
    explicit_code = detect_explicit_language_request(cleaned)
    if explicit_code and explicit_code in SUPPORTED_LANGUAGE_CODES:
        target_obj = LANGUAGE_BY_CODE[explicit_code]
        return LanguageResolution(
            target_language=explicit_code,
            instruction_name=target_obj.instruction_name,
            detected_language=detected_code,
            confidence=confidence,
            rule="explicit request",
        )

    # 3. Check if text is short (< 3 words) or detection confidence is low (< 0.35)
    is_short_or_low_conf = is_short_text(cleaned) or confidence < 0.35

    if not is_short_or_low_conf and detected_code in SUPPORTED_LANGUAGE_CODES:
        target_obj = LANGUAGE_BY_CODE[detected_code]
        return LanguageResolution(
            target_language=detected_code,
            instruction_name=target_obj.instruction_name,
            detected_language=detected_code,
            confidence=confidence,
            rule="detected",
        )

    # 4. Fallback chain: previous user message in chat, else profile default
    fallback_code: str | None = None
    if history:
        for turn in reversed(history):
            if turn.get("role") == "user" and str(turn.get("content", "")).strip():
                prev_text = str(turn.get("content", "")).strip()
                # Check explicit request or detection on previous user message
                prev_explicit = detect_explicit_language_request(prev_text)
                if prev_explicit:
                    fallback_code = prev_explicit
                    break
                prev_detected, prev_conf = detect_language(prev_text)
                if not is_short_text(prev_text) and prev_conf >= 0.35:
                    fallback_code = prev_detected
                    break

    if not fallback_code or fallback_code not in SUPPORTED_LANGUAGE_CODES:
        fallback_code = (
            profile_default_language
            if profile_default_language in SUPPORTED_LANGUAGE_CODES
            else "en"
        )

    target_obj = LANGUAGE_BY_CODE[fallback_code]
    return LanguageResolution(
        target_language=fallback_code,
        instruction_name=target_obj.instruction_name,
        detected_language=detected_code,
        confidence=confidence,
        rule="fallback",
    )


def assemble_translation_messages(
    *,
    text: str,
    target_language: str,
    source_language: str | None = None,
) -> list[dict[str, str]]:
    """Assemble faithful, translation-only messages for the LLM (§ Part 2).

    Rules:
    - Translate the text faithfully; do not add, remove or reinterpret information.
    - Keep [n] citation markers, numbers, units, product names and model numbers unchanged.
    - Output only the translation.
    """
    target_entry = LANGUAGE_BY_CODE.get(target_language) or LANGUAGE_BY_CODE["en"]
    source_entry = LANGUAGE_BY_CODE.get(source_language) if source_language else None

    # Include native script if not already in instruction name (e.g. Simplified Chinese (简体中文))
    if source_entry:
        source_label = (
            f"{source_entry.instruction_name} ({source_entry.native_name})"
            if source_entry.native_name not in source_entry.instruction_name
            else source_entry.instruction_name
        )
        source_clause = f"from {source_label} "
    else:
        source_clause = ""

    target_label = (
        f"{target_entry.instruction_name} ({target_entry.native_name})"
        if target_entry.native_name not in target_entry.instruction_name
        else target_entry.instruction_name
    )

    system_content = (
        f"You are a professional translator. Translate the text faithfully {source_clause}into {target_label}.\n"
        "- Translate the text faithfully; do not add, remove or reinterpret information.\n"
        "- Keep [n] citation markers, numbers, units, product names and model numbers unchanged.\n"
        "- Output only the translation without any preamble, notes, or explanations."
    )
    user_content = (
        f"Translate the following text into {target_label}:\n\n"
        f"{text.strip()}"
    )
    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]
