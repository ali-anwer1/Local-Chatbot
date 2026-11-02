# Multilingual answers: reply in the asker's language, plus on-demand translation

Problem: questions asked in Malay, Mandarin, etc. are answered in English. Cause: nothing in the prompt tells the model which language to answer in, and the knowledge-base passages are in English, so the small model follows the passages. Several other parts of the pipeline also assume English (see Part 1).

Read `app/core/rag.py`, `app/core/service.py`, `app/core/profiles.py`, `app/static/js/chat.js` and `samples/run_eval.py` before starting. Keep the app fully offline. Keep existing tests passing and add tests where noted. Stop after each part and summarise what changed.

---

## Part 1 — Answer in the language the question was asked in

### Supported languages
Define one list in a single place (backend constant, also served to the frontend via an endpoint):
`en` English, `ms` Bahasa Melayu, `zh` 简体中文 (Simplified Chinese), `ta` தமிழ் (Tamil), `id` Bahasa Indonesia, `ja` 日本語, `ko` 한국어, `ar` العربية.
Each entry: code, English name, native name, and the exact name to use in model instructions (e.g. "Bahasa Melayu (Malay)", "Simplified Chinese").

### Language detection (in code, not by the model)
1. Add `lingua-language-detector` to `requirements.txt`. Build one detector restricted to the supported languages above (not all languages), created once at startup.
2. Detect the language of the latest user message only. Rules:
   - If the text is very short (under ~3 words) or detection confidence is low, use the language of the previous user message in this chat; if none, use the profile's default language (new profile field `default_language`, default `en`).
   - Malay and Indonesian are easily confused on short text. If detection returns `id` but confidence over `ms` is not clearly higher, prefer `ms` (Malaysian context).
3. Explicit override wins over detection, in this order:
   a. A per-chat "Answer in" selector in the composer (see UI below), when set to something other than Auto.
   b. An explicit request in the question text ("answer in English", "jawab dalam bahasa Melayu", "用中文回答"). Detect this with a small regex/keyword table for the supported languages; if matched, use that language.
   c. Detected language.

### Prompt changes (`assemble_messages`)
4. Add to the grounding rules: answer in the target language; keep citation markers `[n]` exactly as they are; keep product names, model numbers, units and quoted document titles as written in the source; translate everything else.
5. Append a short final line to the end of the user payload (after the question): `Answer in <instruction name>.` Putting it last matters for small models and keeps the cacheable prefix (system + history) unchanged.

### Refusals
6. The refusal gate refuses in code before calling the model, so it can't rely on the model to translate. Change `refusal_message` on the profile into per-language messages:
   - Keep the existing string as the `en` value (migrate existing profiles on load without losing it).
   - Provide built-in defaults for every supported language; the user can override any of them in Settings.
   - Code-path refusals use the message for the target language.
7. Pass the target-language refusal message in the system prompt (instead of the English one), and update the "model refused" check (currently `full_text.lower().startswith(profile.refusal_message...)`) to compare against the target-language message, falling back to checking all languages' messages.

### Other English-only logic to fix
8. `PURE_GREETING_PATTERN` and `CAPABILITY_QUERY_PATTERN`: add common greetings/capability questions in Malay and Chinese at minimum (e.g. "hai", "helo", "selamat pagi", "terima kasih", "apa yang boleh awak bantu", "你好", "谢谢", "你能帮我什么"). Conversational replies must also follow the target-language rule.
9. `FOLLOWUP_PRONOUN_PATTERN` (used to decide whether to fold history into the retrieval query): add Malay and Chinese equivalents (e.g. "ia", "itu", "ini", "mereka", "nya" suffix, "它", "这个", "那个", "他们").

### Cross-language retrieval check (measure, don't change models)
10. The embedding model (`nomic-embed-text`) is mainly English, so a Malay or Chinese question may score lower against English passages and get refused by the similarity gate even when the answer exists. Do not switch models automatically. Instead:
    - Create `samples/eval_questions_ms.json` and `samples/eval_questions_zh.json`: the same questions as `eval_questions.json` translated into Malay and Simplified Chinese, same expected sources. Leave `expected_keywords` that are numbers/model names unchanged.
    - Extend `run_eval.py` to accept a questions file and an embedding-model override, and to print a summary table per file: pass rate, mean best_score for in-scope questions, number of wrong refusals, and whether the answer came back in the expected language (use the same detector).
    - Print a one-line comparison so the user can decide whether to switch to a multilingual embedding model (e.g. `bge-m3`) via the existing embedding-model setting.

### UI
11. In the composer, next to the Thinking toggle, add a small "Answer in: Auto ▾" selector listing Auto plus the supported languages (native names). Per-chat state; resets to Auto on New chat. Send the choice with each request.
12. Under each answer in Debug mode, show the detected language, confidence, and which rule chose the target language (selector / explicit request / detected / fallback).
13. In Settings (Profile section), add "Default language" (select) and the per-language refusal messages (collapsed under "Refusal message in other languages"). If the info-tooltip system from the earlier prompt exists, add tooltips for these two fields.

### Tests
14. Detection + override precedence; Malay/Indonesian tie-break; short-message fallback; refusal message chosen per language on the code path; model-refusal check works for non-English refusal; greeting patterns match the new Malay/Chinese phrases; the language line is the last line of the user payload; profile migration keeps the existing refusal text.

---

## Part 2 — Translate an answer on demand

15. Add a "Translate" control to each completed assistant answer (in the answer's action row; a small menu listing the supported languages by native name, excluding the answer's current language).
16. Backend: `POST /api/translate` with the answer text, source language, and target language. Call the profile's chat model with a translation-only prompt:
    - Translate the text faithfully; do not add, remove or reinterpret information.
    - Keep `[n]` citation markers, numbers, units, product names and model numbers unchanged.
    - Output only the translation.
    Use temperature 0.1, `think: false`, and stream the result. No retrieval and no refusal gate (the content has already been grounded).
17. Show the translation below the original answer in a collapsible block labelled with the language (e.g. "Bahasa Melayu ▾"), with "Show original"/"Hide" and a copy button. Citations in the translation link to the same Sources list as the original. Multiple translations of the same answer can be open; cache each per answer so re-selecting doesn't call the model again.
18. Saved chats and Export include translations that were generated, labelled by language.
19. Tests: translation endpoint builds the right prompt, never calls retrieval, preserves citation markers in a mocked response; cache prevents a second backend call.

---

## Done when
- Asking in Malay or Mandarin returns an answer in that language with working citations; "answer in English" inside a Malay question returns English; the composer selector overrides both.
- Refusals (code path and model path) appear in the target language.
- `run_eval.py` runs on the English, Malay and Chinese question files and prints the comparison table.
- Any answer can be translated into the supported languages from its action row.
- All tests pass.
