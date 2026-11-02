"""RAG retrieval, conversational intent handling, refusal gate, citation filtering, and prompt assembly (§4.3)."""

from __future__ import annotations

from collections.abc import Generator, Iterator
from dataclasses import dataclass, field
import re
import time
from typing import Any

from app.core.availability import ModelAvailabilityManager
from app.core.backends.base import LLMBackend
from app.core.errors import ValidationError
from app.core.languages import (
    DEFAULT_REFUSAL_MESSAGES,
    LANGUAGE_BY_CODE,
    LanguageResolution,
    resolve_target_language,
)
from app.core.profiles import Profile
from app.core.vectorstore import RetrievedChunk, VectorStore


FIXED_GROUNDING_RULES = (
    "Answer only using the numbered context passages. Cite passages as [n]. "
    "If the passages do not contain the answer, reply exactly with the refusal message "
    "and briefly state what information is missing. Do not use outside knowledge."
)

CONVERSATIONAL_GROUNDING_RULES = (
    "Respond warmly, naturally, and concisely in character using ONLY your persona above "
    "and the Knowledge Base Overview provided below. Explain what topics and uploaded documents "
    "you can assist with, and optionally suggest 2-3 example questions based on those documents. "
    "Do NOT answer general world-knowledge trivia, unrelated tasks, or claim knowledge outside "
    "the listed documents. Do NOT include bracketed citation numbers like [1]."
)

FOLLOWUP_PRONOUN_PATTERN = re.compile(
    r"(?:\b(it|its|they|them|their|theirs|this|that|these|those|he|him|his|she|her|hers|there|same|former|latter|"
    r"ia|dia|mereka|ini|itu|tersebut|berkenaan|\w+nya)\b|"
    r"[它牠祂他她][们們]?|[这這那][个個些]|[其该該])",
    re.IGNORECASE,
)

PURE_GREETING_PATTERN = re.compile(
    r"^(?:"
    r"hi+|hello+|hey+|howdy|greetings|yo|"
    r"(?:hi|hello|hey)\s+(?:there|bot|assistant)|"
    r"good\s+(?:morning|afternoon|evening|day)|"
    r"how\s+are\s+you(?:\s+doing|\s+today)?|how'?s\s+it\s+going|"
    r"thanks?(?:\s+a\s+lot|\s+so\s+much|\s+very\s+much)?|"
    r"thank\s+you(?:\s+so\s+much|\s+very\s+much|\s+a\s+lot)?|"
    r"(?:many\s+)?thanks|appreciate\s+it|great\s+thanks|awesome\s+thanks|"
    r"bye|goodbye|see\s+you(?:\s+later)?|have\s+a\s+(?:good|great)\s+day|"
    r"hai+|helo+|salam|assalam(?:ualaikum)?|"
    r"selamat\s+(?:pagi|petang|tengah\s+hari|malam|sejahtera|tinggal|jalan)|"
    r"terima\s+kasih(?:\s+banyak)?|makasih|tq|apa\s+khabar|"
    r"你好|您好|早安|早上好|下午好|晚上好|嗨|哈啰|哈喽|"
    r"谢谢|多谢|感谢|非常感谢|十分感谢|再见|拜拜|明天见"
    r")[!.,?\s]*$",
    re.IGNORECASE,
)

CAPABILITY_QUERY_PATTERN = re.compile(
    r"^(?:(?:hi|hello|hey|hai|helo|你好)[,!.\s]+)?(?:"
    r"what\s+can\s+you\s+(?:help(?:\s+me)?(?:\s+with)?|do|answer|assist(?:\s+me)?(?:\s+with)?)|"
    r"how\s+can\s+you\s+(?:help|assist)(?:\s+me)?|"
    r"who\s+are\s+you|what\s+are\s+you|tell\s+me\s+about\s+yourself|introduce\s+yourself|"
    r"what\s+is\s+your\s+(?:role|purpose|job|function|scope)|"
    r"what\s+(?:do\s+you\s+know(?:\s+about)?|knowledge\s+do\s+you\s+have)|"
    r"what\s+(?:topics?|subjects?|areas?|things?|information)\s+(?:do\s+you\s+(?:cover|know|have)|can\s+you\s+(?:help\s+with|answer))|"
    r"what\s+(?:documents?|files?|sources?|manuals?|pdfs?)\s+(?:do\s+you\s+have|are\s+(?:in|uploaded|available|indexed))|"
    r"(?:list|show|summarize|describe)\s+(?:your|the)\s+(?:documents?|files?|knowledge\s*base|topics?|capabilities)|"
    r"what(?:'s|\s+is)\s+in\s+(?:your|the|this)\s+knowledge\s*base|"
    r"what\s+(?:kind\s+of\s+)?questions?\s+can\s+i\s+ask(?:\s+you)?|"
    r"what\s+can\s+i\s+ask(?:\s+you)?(?:\s+about)?|"
    r"help|help\s+me|"
    r"apa\s+(?:yang\s+)?(?:(?:boleh|dapat)\s+(?:awak|anda|kamu)|(?:awak|anda|kamu)\s+(?:boleh|dapat))\s+(?:bantu|tolong|buat|jawab)|"
    r"bagaimana\s+(?:awak|anda|kamu)\s+(?:boleh\s+)?(?:bantu|tolong)(?:\s+saya)?|"
    r"siapa\s+(?:awak|anda|kamu)|"
    r"apa\s+peranan\s+(?:awak|anda|kamu)|"
    r"apa\s+(?:dokumen|fail|topik)\s+yang\s+(?:ada|dimuat\s+naik|tersedia)|"
    r"bantuan|tolong(?:\s+saya)?|"
    r"(?:你能|你可以|您能|您可以)(?:帮我|为我)?(?:做什么|做些什么|干什么)|"
    r"你能帮我什么|你可以帮我什么|你是谁|你叫什么|介绍(?:一下)?你自己|"
    r"你有什么功能|你的功能是什么|你的职责是什么|你的作用是什么|"
    r"你(?:知道|了解|掌握)(?:什么|哪些内容|哪些知识)|"
    r"知识库(?:里|中)?有(?:什么|哪些)(?:文档|文件|资料|内容)|"
    r"(?:列出|展示|总结|介绍)(?:你的|知识库的)?(?:文档|文件|内容|功能)|"
    r"帮助|求助|帮帮我"
    r")[!.,?\s]*$",
    re.IGNORECASE,
)

CITATION_BRACKET_PATTERN = re.compile(
    r"\[(?:(?:passage|source|doc|ref)s?\s*)?(\d+(?:\s*(?:,|-|–)\s*\d+)*)\]",
    re.IGNORECASE,
)


def classify_conversational_intent(question: str) -> str | None:
    """Detect pure greetings/pleasantries or self-referential capability/overview queries.

    Returns 'greeting', 'capability', or None (for normal queries that must pass the RAG gate).
    """
    normalized = re.sub(r"\s+", " ", question.strip())
    if not normalized:
        return None
    if PURE_GREETING_PATTERN.match(normalized):
        return "greeting"
    if CAPABILITY_QUERY_PATTERN.match(normalized):
        return "capability"
    return None


def extract_cited_indices(answer_text: str, max_index: int) -> list[int]:
    """Extract 1-based citation numbers [n] actually referenced in the generated answer."""
    if not answer_text or max_index <= 0:
        return []

    seen: set[int] = set()
    ordered: list[int] = []

    for match in CITATION_BRACKET_PATTERN.finditer(answer_text):
        expr = match.group(1)
        for part in expr.split(","):
            token = part.strip()
            if not token:
                continue
            if "-" in token or "–" in token:
                bounds = re.split(r"[-–]", token, maxsplit=1)
                if len(bounds) == 2 and bounds[0].strip().isdigit() and bounds[1].strip().isdigit():
                    start_i = int(bounds[0].strip())
                    end_i = int(bounds[1].strip())
                    for idx in range(min(start_i, end_i), max(start_i, end_i) + 1):
                        if 1 <= idx <= max_index and idx not in seen:
                            seen.add(idx)
                            ordered.append(idx)
            elif token.isdigit():
                idx = int(token)
                if 1 <= idx <= max_index and idx not in seen:
                    seen.add(idx)
                    ordered.append(idx)

    return ordered


def build_retrieval_query(
    question: str, history: list[dict[str, str]] | None = None
) -> str:
    """Build retrieval query; prepend previous user message for short/pronoun follow-ups (§4.3)."""
    cleaned = question.strip()
    if not cleaned:
        raise ValidationError("Question cannot be empty.")

    if not history:
        return cleaned

    prev_user_msg: str | None = None
    for msg in reversed(history):
        if msg.get("role") == "user" and str(msg.get("content", "")).strip():
            candidate = str(msg["content"]).strip()
            # Don't prepend a pure greeting like "Hi" onto the next real question
            if classify_conversational_intent(candidate) is None:
                prev_user_msg = candidate
                break

    if not prev_user_msg:
        return cleaned

    words = cleaned.split()
    is_short = len(words) <= 6 or len(cleaned) < 40
    has_pronoun = bool(FOLLOWUP_PRONOUN_PATTERN.search(cleaned))

    if is_short or has_pronoun:
        return f"{prev_user_msg} {cleaned}"
    return cleaned


def format_source_label(chunk: RetrievedChunk) -> str:
    """Format human-readable source provenance for prompt context and citations."""
    if chunk.location_label:
        return f"{chunk.source_file}, {chunk.location_label}"
    return chunk.source_file


def _trim_history(
    history: list[dict[str, str]] | None, max_messages: int
) -> list[dict[str, str]]:
    if not history or max_messages <= 0:
        return []
    trimmed: list[dict[str, str]] = []
    for turn in history[-max_messages:]:
        role = turn.get("role", "")
        content = str(turn.get("content", "")).strip()
        if role in {"user", "assistant"} and content:
            trimmed.append({"role": role, "content": content})
    return trimmed


def _meta_event(
    *,
    profile: Profile,
    effective_chat_model: str,
    fallback_active: bool,
    fallback_banner: str | None,
    retrieval_query: str,
    best_score: float,
    refused: bool = False,
    conversational: bool = False,
    intent: str | None = None,
    refusal_reason: str | None = None,
    candidate_sources: list[dict[str, Any]] | None = None,
    debug_chunks: list[dict[str, Any]] | None = None,
    language_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    data: dict[str, Any] = {
        "refused": refused,
        "conversational": conversational,
        "refusal_reason": refusal_reason,
        "best_score": round(best_score, 4),
        "threshold": profile.similarity_threshold,
        "retrieval_query": retrieval_query,
        "sources": [],
        "candidate_sources": candidate_sources if candidate_sources is not None else [],
        "debug_chunks": debug_chunks if debug_chunks is not None else [],
        "configured_chat_model": profile.chat_model,
        "effective_chat_model": effective_chat_model,
        "fallback_active": fallback_active,
        "fallback_banner": fallback_banner,
    }
    if intent is not None:
        data["intent"] = intent
    if language_meta:
        data["language"] = language_meta
    return {"event": "meta", "data": data}


def assemble_rag_messages(
    *,
    profile: Profile,
    question: str,
    retrieved_chunks: list[RetrievedChunk],
    history: list[dict[str, str]] | None = None,
    target_language: str = "en",
) -> list[dict[str, str]]:
    """Assemble system preamble + fixed grounding rules, trimmed history, and numbered context (§4.3)."""
    lang_entry = LANGUAGE_BY_CODE.get(target_language) or LANGUAGE_BY_CODE["en"]
    target_refusal = profile.get_refusal_message(target_language)

    preamble_part = profile.preamble.strip()
    system_parts: list[str] = []
    if preamble_part:
        system_parts.append(preamble_part)

    grounding_rules = (
        "Answer only using the numbered context passages. Cite passages as [n]. "
        "If the passages do not contain the answer, reply exactly with the refusal message "
        "and briefly state what information is missing. Do not use outside knowledge. "
        f"Answer in {lang_entry.instruction_name}. Keep citation markers [n] exactly as they are. "
        "Keep product names, model numbers, units, and quoted document titles as written in the source; "
        "translate everything else."
    )
    system_parts.append(
        f"{grounding_rules}\nRefusal message: \"{target_refusal}\""
    )
    system_content = "\n\n".join(system_parts)

    messages: list[dict[str, str]] = [{"role": "system", "content": system_content}]
    messages.extend(_trim_history(history, profile.history_turns * 2))

    context_blocks: list[str] = []
    for idx, chunk in enumerate(retrieved_chunks, start=1):
        label = format_source_label(chunk)
        context_blocks.append(f"[{idx}] (Source: {label})\n{chunk.text}")

    joined_context = "\n\n".join(context_blocks)
    user_payload = (
        f"Context passages:\n{joined_context}\n\n"
        f"Question: {question.strip()}\n"
        f"Answer in {lang_entry.instruction_name}."
    )
    messages.append({"role": "user", "content": user_payload})
    return messages


def assemble_conversational_messages(
    *,
    profile: Profile,
    question: str,
    documents_overview: list[dict[str, Any]],
    sample_snippets: list[dict[str, Any]],
    history: list[dict[str, str]] | None = None,
    target_language: str = "en",
) -> list[dict[str, str]]:
    """Assemble bounded prompt for greetings and 'What can you help me with?' queries."""
    lang_entry = LANGUAGE_BY_CODE.get(target_language) or LANGUAGE_BY_CODE["en"]
    persona = profile.preamble.strip() or f"You are {profile.name}, a local knowledge base assistant."
    conversational_rules = (
        f"{CONVERSATIONAL_GROUNDING_RULES} "
        f"Answer in {lang_entry.instruction_name}."
    )
    system_content = (
        f"Assistant Name: {profile.name}\n"
        f"Persona / Preamble: {persona}\n\n"
        f"{conversational_rules}"
    )

    indexed_docs = [d for d in documents_overview if d.get("status") == "indexed"]
    if not indexed_docs:
        kb_summary = (
            "Knowledge Base Status: No documents have been uploaded to this profile yet. "
            "Let the user know your intended role from your persona and invite them to upload "
            "PDF, DOCX, Markdown, or TXT documents using the Settings & Knowledge Base icon in the top-right."
        )
    else:
        doc_lines = [
            f"- {d['filename']} ({d.get('chunk_count', 0)} indexed chunks)"
            for d in indexed_docs[:25]
        ]
        if len(indexed_docs) > 25:
            doc_lines.append(f"- ...and {len(indexed_docs) - 25} more documents")
        snippet_lines = []
        for s in sample_snippets[:6]:
            loc = f" ({s['section']})" if s.get("section") else (f" (Page {s['page']})" if s.get("page") else "")
            snippet_lines.append(f"  * {s['source_file']}{loc}: {s['preview']}")

        kb_summary = "Uploaded Documents in Knowledge Base:\n" + "\n".join(doc_lines)
        if snippet_lines:
            kb_summary += "\n\nSample Topic Excerpts:\n" + "\n".join(snippet_lines)

    messages: list[dict[str, str]] = [{"role": "system", "content": system_content}]
    messages.extend(_trim_history(history, min(4, profile.history_turns * 2)))

    messages.append(
        {
            "role": "user",
            "content": f"{kb_summary}\n\nUser message: {question.strip()}\nAnswer in {lang_entry.instruction_name}.",
        }
    )
    return messages


def _build_timings(
    *,
    t_start: float,
    embed_ms: float | None = None,
    retrieval_ms: float | None = None,
    time_to_first_token_ms: float | None = None,
    chat_stats: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble per-response pipeline and Ollama generation timing breakdown."""
    total_ms = round((time.perf_counter() - t_start) * 1000.0, 2)
    timings: dict[str, Any] = {
        "embed_ms": embed_ms,
        "retrieval_ms": retrieval_ms,
        "time_to_first_token_ms": time_to_first_token_ms,
        "total_ms": total_ms,
    }
    if chat_stats:
        for key in (
            "prompt_eval_count",
            "prompt_eval_duration",
            "eval_count",
            "eval_duration",
        ):
            if key in chat_stats and chat_stats[key] is not None:
                timings[key] = chat_stats[key]

        prompt_dur = chat_stats.get("prompt_eval_duration")
        if isinstance(prompt_dur, (int, float)) and prompt_dur >= 0:
            timings["prompt_eval_ms"] = round(prompt_dur / 1_000_000.0, 2)

        eval_dur = chat_stats.get("eval_duration")
        eval_cnt = chat_stats.get("eval_count")
        if isinstance(eval_dur, (int, float)) and eval_dur >= 0:
            timings["eval_ms"] = round(eval_dur / 1_000_000.0, 2)
            if isinstance(eval_cnt, (int, float)) and eval_dur > 0:
                timings["tokens_per_second"] = round(
                    eval_cnt / (eval_dur / 1_000_000_000.0), 2
                )
    return timings


@dataclass
class RAGResponse:
    """Complete result of a non-streaming RAG query (useful for CLI/tests/eval)."""

    answer: str
    refused: bool
    refusal_reason: str | None
    best_score: float
    threshold: float
    retrieval_query: str
    sources: list[dict[str, Any]] = field(default_factory=list)
    debug_chunks: list[dict[str, Any]] = field(default_factory=list)
    effective_chat_model: str = ""
    fallback_active: bool = False
    fallback_banner: str | None = None
    conversational: bool = False
    timings: dict[str, Any] = field(default_factory=dict)
    target_language: str = "en"
    detected_language: str = "en"
    language_confidence: float = 0.0
    language_rule: str = "fallback"


class RAGEngine:
    """Executes retrieval, code-enforced refusal gating, and grounded LLM generation (§4.3)."""

    def __init__(
        self,
        *,
        backend: LLMBackend,
        vectorstore: VectorStore,
        availability: ModelAvailabilityManager,
    ) -> None:
        self.backend = backend
        self.vectorstore = vectorstore
        self.availability = availability

    def _stream_chat_tokens(
        self,
        *,
        messages: list[dict[str, str]],
        model: str,
        options: dict[str, Any],
        t_start: float,
    ) -> Generator[dict[str, Any], None, tuple[str, float | None]]:
        collected: list[str] = []
        ttft_ms: float | None = None
        for token in self.backend.chat(
            messages=messages,
            model=model,
            options=options,
            stream=True,
        ):
            if ttft_ms is None:
                ttft_ms = round((time.perf_counter() - t_start) * 1000.0, 2)
            collected.append(token)
            yield {"event": "token", "data": {"text": token}}
        return "".join(collected).strip(), ttft_ms

    def stream_query(
        self,
        *,
        profile: Profile,
        question: str,
        history: list[dict[str, str]] | None = None,
        runtime_options: dict[str, Any] | None = None,
        documents_overview: list[dict[str, Any]] | None = None,
        requested_language: str | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Execute full RAG flow and yield structured events ('status', 'meta', 'token', 'done').

        Guarantees that when the refusal gate fails on factual queries, backend.chat() is NEVER called.
        """
        t_start = time.perf_counter()
        cleaned_question = question.strip()
        if not cleaned_question:
            raise ValidationError("Message cannot be empty.")

        lang_res = resolve_target_language(
            latest_user_message=cleaned_question,
            history=history,
            profile_default_language=profile.default_language,
            requested_language=requested_language,
        )
        target_lang_code = lang_res.target_language
        target_refusal_message = profile.get_refusal_message(target_lang_code)
        lang_meta = lang_res.to_dict()

        (
            effective_chat_model,
            effective_embed_model,
            fallback_active,
            fallback_banner,
        ) = self.availability.require_ready_for_chat(profile, self.vectorstore)

        use_thinking = bool(
            runtime_options.get("think")
            if (runtime_options and "think" in runtime_options)
            else profile.enable_thinking
        )
        merged_options: dict[str, Any] = {
            "temperature": profile.temperature,
            "think": use_thinking,
        }
        if runtime_options:
            merged_options.update(runtime_options)
            merged_options["think"] = use_thinking

        # --- CONVERSATIONAL / CAPABILITY INTENT PATH ---
        conv_intent = classify_conversational_intent(cleaned_question)
        if conv_intent is not None:
            samples = (
                self.vectorstore.peek_samples(profile.id, limit=6)
                if conv_intent == "capability"
                else []
            )
            conv_messages = assemble_conversational_messages(
                profile=profile,
                question=cleaned_question,
                documents_overview=documents_overview or [],
                sample_snippets=samples,
                history=history,
                target_language=target_lang_code,
            )
            gen_label = (
                f"Thinking with {effective_chat_model}…"
                if use_thinking
                else f"Preparing response with {effective_chat_model}…"
            )
            yield {
                "event": "status",
                "data": {"stage": "generating", "label": gen_label},
            }
            yield _meta_event(
                profile=profile,
                effective_chat_model=effective_chat_model,
                fallback_active=fallback_active,
                fallback_banner=fallback_banner,
                retrieval_query=cleaned_question,
                best_score=1.0,
                conversational=True,
                intent=conv_intent,
                language_meta=lang_meta,
            )

            conv_text, ttft_ms = yield from self._stream_chat_tokens(
                messages=conv_messages,
                model=effective_chat_model,
                options=merged_options,
                t_start=t_start,
            )
            timings = _build_timings(
                t_start=t_start,
                time_to_first_token_ms=ttft_ms,
                chat_stats=self.backend.get_last_chat_stats(),
            )
            yield {
                "event": "done",
                "data": {
                    "full_text": conv_text,
                    "refused": False,
                    "conversational": True,
                    "refusal_reason": None,
                    "sources": [],
                    "timings": timings,
                    "language": lang_meta,
                },
            }
            return

        # --- KNOWLEDGE BASE RETRIEVAL & REFUSAL GATE PATH ---
        retrieval_query = build_retrieval_query(cleaned_question, history)

        inspection = self.vectorstore.inspect_collection(profile.id)
        retrieved: list[RetrievedChunk] = []
        embed_ms: float | None = None
        retrieval_ms: float | None = None
        if inspection.exists and inspection.chunk_count > 0:
            t_embed_start = time.perf_counter()
            query_vectors = self.backend.embed([retrieval_query], model=effective_embed_model)
            embed_ms = round((time.perf_counter() - t_embed_start) * 1000.0, 2)
            query_vec = query_vectors[0]

            t_retrieval_start = time.perf_counter()
            retrieved = self.vectorstore.query(
                profile_id=profile.id,
                query_embedding=query_vec,
                expected_embedding_model=profile.embedding_model,
                top_k=profile.top_k,
            )
            retrieval_ms = round((time.perf_counter() - t_retrieval_start) * 1000.0, 2)

        best_score = retrieved[0].score if retrieved else 0.0
        candidate_sources = [
            c.to_dict(citation_index=i + 1) for i, c in enumerate(retrieved)
        ]

        # --- REFUSAL GATE IN CODE (before any LLM call!) ---
        if not retrieved or best_score < profile.similarity_threshold:
            reason = (
                f"No relevant content found in this bot's knowledge base. "
                f"Best match scored {best_score:.2f}, threshold is {profile.similarity_threshold:.2f}."
            )
            refusal_text = f"{target_refusal_message.strip()} {reason}"
            yield _meta_event(
                profile=profile,
                effective_chat_model=effective_chat_model,
                fallback_active=fallback_active,
                fallback_banner=fallback_banner,
                retrieval_query=retrieval_query,
                best_score=best_score,
                refused=True,
                refusal_reason=reason,
                debug_chunks=candidate_sources,
                language_meta=lang_meta,
            )
            yield {"event": "token", "data": {"text": refusal_text}}
            timings = _build_timings(
                t_start=t_start,
                embed_ms=embed_ms,
                retrieval_ms=retrieval_ms,
            )
            yield {
                "event": "done",
                "data": {
                    "full_text": refusal_text,
                    "refused": True,
                    "conversational": False,
                    "refusal_reason": reason,
                    "sources": [],
                    "timings": timings,
                    "language": lang_meta,
                },
            }
            return

        # Gate passed — assemble grounded context and stream from LLM
        messages = assemble_rag_messages(
            profile=profile,
            question=cleaned_question,
            retrieved_chunks=retrieved,
            history=history,
            target_language=target_lang_code,
        )

        gen_stage_label = (
            f"Thinking over {len(retrieved)} passage(s) with {effective_chat_model}…"
            if use_thinking
            else f"Generating answer from {len(retrieved)} passage(s)…"
        )
        yield {
            "event": "status",
            "data": {"stage": "generating", "label": gen_stage_label},
        }
        yield _meta_event(
            profile=profile,
            effective_chat_model=effective_chat_model,
            fallback_active=fallback_active,
            fallback_banner=fallback_banner,
            retrieval_query=retrieval_query,
            best_score=best_score,
            candidate_sources=candidate_sources,
            debug_chunks=candidate_sources,
            language_meta=lang_meta,
        )

        full_text, ttft_ms = yield from self._stream_chat_tokens(
            messages=messages,
            model=effective_chat_model,
            options=merged_options,
            t_start=t_start,
        )
        timings = _build_timings(
            t_start=t_start,
            embed_ms=embed_ms,
            retrieval_ms=retrieval_ms,
            time_to_first_token_ms=ttft_ms,
            chat_stats=self.backend.get_last_chat_stats(),
        )

        # Check target language refusal first, then all languages
        prompt_refused = full_text.lower().startswith(target_refusal_message.strip().lower())
        if not prompt_refused:
            for other_ref in profile.refusal_messages.values():
                if other_ref and full_text.lower().startswith(other_ref.strip().lower()):
                    prompt_refused = True
                    break

        # Filter returned sources strictly to passages actually cited as [n] in the answer
        cited_indices = set(
            extract_cited_indices(full_text, max_index=len(candidate_sources))
        )
        used_sources = (
            []
            if prompt_refused
            else [s for s in candidate_sources if int(s["index"]) in cited_indices]
        )
        enriched_debug = [
            {**chunk_dict, "cited": int(chunk_dict["index"]) in cited_indices}
            for chunk_dict in candidate_sources
        ]

        yield {
            "event": "done",
            "data": {
                "full_text": full_text,
                "refused": prompt_refused,
                "conversational": False,
                "refusal_reason": "Context passages did not contain the requested answer."
                if prompt_refused
                else None,
                "sources": used_sources,
                "debug_chunks": enriched_debug,
                "timings": timings,
                "language": lang_meta,
            },
        }

    def query(
        self,
        *,
        profile: Profile,
        question: str,
        history: list[dict[str, str]] | None = None,
        runtime_options: dict[str, Any] | None = None,
        documents_overview: list[dict[str, Any]] | None = None,
        requested_language: str | None = None,
    ) -> RAGResponse:
        """Synchronous convenience wrapper around stream_query for tests and CLI tools."""
        meta_data: dict[str, Any] = {}
        done_data: dict[str, Any] = {}
        tokens: list[str] = []

        for item in self.stream_query(
            profile=profile,
            question=question,
            history=history,
            runtime_options=runtime_options,
            documents_overview=documents_overview,
            requested_language=requested_language,
        ):
            ev = item["event"]
            if ev == "meta":
                meta_data = item["data"]
            elif ev == "token":
                tokens.append(str(item["data"].get("text", "")))
            elif ev == "done":
                done_data = item["data"]

        answer = str(done_data.get("full_text") or "".join(tokens)).strip()
        lang_info = done_data.get("language") or meta_data.get("language") or {}
        return RAGResponse(
            answer=answer,
            refused=bool(done_data.get("refused", meta_data.get("refused", False))),
            refusal_reason=done_data.get("refusal_reason") or meta_data.get("refusal_reason"),
            best_score=float(meta_data.get("best_score", 0.0)),
            threshold=float(meta_data.get("threshold", profile.similarity_threshold)),
            retrieval_query=str(meta_data.get("retrieval_query", question)),
            sources=list(done_data.get("sources", [])),
            debug_chunks=list(done_data.get("debug_chunks", meta_data.get("debug_chunks", []))),
            effective_chat_model=str(meta_data.get("effective_chat_model", profile.chat_model)),
            fallback_active=bool(meta_data.get("fallback_active", False)),
            fallback_banner=meta_data.get("fallback_banner"),
            conversational=bool(done_data.get("conversational", meta_data.get("conversational", False))),
            timings=dict(done_data.get("timings", {})),
            target_language=str(lang_info.get("target_language", "en")),
            detected_language=str(lang_info.get("detected_language", "en")),
            language_confidence=float(lang_info.get("language_confidence", 0.0)),
            language_rule=str(lang_info.get("language_rule", "fallback")),
        )
