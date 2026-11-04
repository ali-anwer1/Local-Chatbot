# Review follow-up — bug fixes and excess-code cleanup (Version 1)

You are changing an existing, working app. Read `README.md` and `prompts/02`–`04` first so you know why things exist. This task fixes bugs found in review and removes dead or duplicated code. **Do not add features.** User-visible behaviour must not change except where an item says so.

## Ground rules
1. Work one phase at a time. At the start of each phase, list the files you will touch. At the end, stop and report: what changed, line counts before/after per file, and test results. Wait for "continue".
2. Run `python -m pytest -q` before Phase 0 and after every phase. A phase is done only when all tests pass. If you change a test, say which and why.
3. Do not touch `data/`, `samples/docs/`, `config.yaml`, `styles.css` (except deleting the two unused selectors in 3f).
4. Keep the hard constraints: bind to 127.0.0.1 only, fully offline, vanilla JS ES modules with no build step, no new dependencies.
5. Keep `RAGEngine.query()` and `ChatbotService` public methods working (used by `tests/` and `samples/run_eval.py`), except the removals listed below.
6. Small edits. Don't rename or reformat code you are not otherwise changing.

---

## Phase 0 — Baseline
Run the tests, record pass count and line counts for every file in `app/`. Change nothing.

## Phase 1 — Bug fixes

**1a. Language debug fields never display (backend/frontend key mismatch).**
`LanguageResolution.to_dict()` (`app/core/languages.py`) emits `target_language, instruction_name, detected_language, confidence, rule`. But `app/static/js/chat.js` `createDebugElement` reads `language_confidence`, `language_rule`, `language_name`, and `RAGEngine.query()` (`app/core/rag.py`, end of file) reads `language_confidence` / `language_rule`. Result: Debug always shows confidence "—" and rule "fallback", and `RAGResponse.language_rule` is always "fallback".
Fix: use the backend names (`confidence`, `rule`, `instruction_name`) in both places. Add a test that `RAGEngine.query()` returns the real rule (e.g. `"selector"` when `requested_language="ms"`).

**1b. `stream_translate` crashes without `profile_id` and ignores model fallback.**
In `app/core/service.py`, the no-`profile_id` branch does `profiles[0]["chat_model"]` on a `Profile` dataclass → `TypeError`. It also uses `profile.chat_model` directly, so translation fails when the chat model is missing but fallback is active (chat itself works in that state).
Fix: make `profile_id` required in `TranslateRequest` and `stream_translate` (the frontend always sends it), delete the no-profile branch, and resolve the model with `self.availability.require_ready_for_chat(profile, self.vectorstore)[0]`. Move the function-local imports in `stream_translate` and `list_languages` to module level. Add a test for the fallback case.

**1c. Code-path refusals are mixed-language.**
In `RAGEngine.stream_query`, the gate builds `refusal_text = f"{target_refusal_message} {reason}"`, where `reason` is an English sentence ("No relevant content found… Best match scored…"). A Malay question gets a Malay refusal followed by English. This text is also sent back as conversation history on later turns.
Fix: the shown/streamed text and `full_text` are the target-language refusal message only. Keep `reason` in `refusal_reason` (meta and done events) so Debug still shows the score. Update affected tests.

**1d. Explicit language-request regexes are too broad.**
In `EXPLICIT_LANGUAGE_REQUEST_PATTERNS`: the `zh` pattern matches a bare `中文` anywhere (all other parts are optional); `ms` matches a bare `dalam bahasa melayu`; `id` a bare `dalam bahasa indonesia`; `ja` a bare `日本語で`; `en` matches `use english`. So "Is the manual available in 中文?" or "Adakah manual ini ada dalam bahasa Melayu?" force the answer language.
Fix: every pattern must require an answer/reply verb (answer/reply/respond/write; jawab/balas/tulis; 回答/回复/作答; 答えて/回答して; 대답해/답변해; أجب). Add tests for both the positive cases already covered and these negative cases.

**1e. Follow-up detection treats every Chinese question as a follow-up.**
`build_retrieval_query` in `rag.py` uses `len(cleaned.split()) <= 6`, which is always true for CJK text (no spaces), so the previous question is always prepended. `FOLLOWUP_PRONOUN_PATTERN` also matches the single characters `其|该|該`, which appear in ordinary Chinese questions.
Fix: use `is_short_text()` from `languages.py` (it already handles CJK) for the length check, and remove the bare `[其该該]` alternative. Add a test: a full-length Chinese question with history is not prefixed.

**1f. Per-run chat stats leak between requests.**
`OllamaBackend` stores stats in both a thread-local and `self._last_chat_stats`, and `get_last_chat_stats()` falls back to the instance value. Starlette runs sync generators in a threadpool and may resume them on different threads, so the thread-local is unreliable and the fallback can return another request's stats.
Fix: remove `_last_chat_stats`, `_thread_local`, `_set_last_chat_stats` and `get_last_chat_stats` (backend and `LLMBackend` base). Give `chat()` an optional `stats: dict | None = None` argument that it fills from the final `done` chunk. In `RAGEngine._stream_chat_tokens`, pass a fresh dict and return it with the text. Update `tests/conftest.py`'s mock backend.

**1g. Settings history misses refusal-message edits in other languages.**
`describe_profile_changes` has no entry for `refusal_messages`, so editing only a non-English refusal logs "No parameter changes". Add `("refusal_messages", "Refusal messages", _fmt_label_only)`. Update the pinned-output test.

## Phase 2 — Backend excess (no behaviour change)

**2a. Duplicate atomic writer.** `profiles._atomic_write_json` is a copy of `config.atomic_write_json`. Delete the copy, import the one in `config.py`, and update `tests/test_profiles_and_docs.py`.

**2b. One refusal representation.** `Profile` stores both `refusal_message` (str) and `refusal_messages` (dict), kept in sync in `validate()`, `with_updates()` and `from_dict()`, and the frontend sends both.
- Keep only `refusal_messages`. In `from_dict`, migrate a legacy `refusal_message` into `refusal_messages["en"]` (existing profiles must keep their text; keep the migration test).
- Remove `refusal_message` from `Profile`, `_PROFILE_MUTABLE_COERCERS`, `ProfilePayload`, `_PROFILE_CHANGE_SPECS` (1g's entry covers it) and `collectSettingsPayload` / `renderSettingsForm` in `settings.js` (the English field just writes `refusal_messages.en`).
- Merge `_coerce_refusal_messages` and the merging loop in `from_dict` into one function.
- `validate()` must only validate. Move the default-filling it does today (default_language reset, refusal defaults) into the coercer / `from_dict`.

**2c. Embedding device / keep-alive plumbing.** `OllamaBackend` keeps its own `_embedding_device` / `_keep_alive` copies, property setters (the `embedding_device` setter writes to disk via `AppSettings.update`), and `bind_app_settings()`; `ChatbotService.__init__` does a `hasattr` dance to bind it.
- `OllamaBackend.__init__` takes `app_settings: AppSettings` (required) and reads `embedding_device` and `keep_alive` from it. Delete the shadow fields, both setters, `bind_app_settings`, and the duplicated `VALID_EMBEDDING_DEVICES` check.
- Remove the `hasattr` block in `ChatbotService.__init__`.
- In `AppSettings`, validate `embedding_device` in one private helper used by `__init__`, `update` and `from_config`.

**2d. Dead backend code.** Delete:
- `OllamaBackend._raw_capabilities` (written, never read) and the third element of `_parse_capabilities`' return.
- `model_info()` (abstract method, Ollama implementation, mock in `conftest.py`) — nothing calls it.
- The non-streaming chat path: `_non_stream_chat` and the `stream` parameter of `chat()`. Convert the three `stream=False` tests in `tests/test_availability_and_api.py` to the streaming path.

**2e. Dead API surface left from prompt 02's one-version grace period.** Delete `DELETE /api/profiles/{id}/history`, `ChatbotService.clear_history`, `ChatRequest.session_id`, and the ignored `session_id` query parameter on `GET /chats/{chat_id}`, plus the tests that pin them. Also delete `GET /api/models` and `ChatbotService.list_models` (the frontend never calls them; `/api/health` already returns the same model lists) — unless you find a caller I missed, in which case keep it and say so.

**2f. `rag.py` duplication.**
- `assemble_rag_messages` repeats `FIXED_GROUNDING_RULES` inline. Build the system text from the constant plus the language sentences.
- `_build_timings` re-lists the stat keys; import `OLLAMA_CHAT_STAT_KEYS` from the backend (or move it to `base.py`).
- The `meta` event sends `candidate_sources` and `debug_chunks` with identical content, plus a `sources` key that is always `[]`. Send `candidate_sources` once; drop `sources` from meta. Update `chat.js` (`meta` handler) and `RAGEngine.query()` to match. The `done` event keeps `sources` and `debug_chunks` (with `cited` flags).

**2g. Small items.**
- `ingestion.py`: in `parse_markdown` and `parse_docx` both branches of the heading `if` append to the buffer — keep the append outside the `if`. In `_split_Into_atomic_pieces`, the `not separators` and `sep == ""` branches do the same thing; keep one. Rename to `_split_into_atomic_pieces`.
- `vectorstore.peek_samples`: the first `if … continue` is dead given the second check; simplify to one-sample-per-file.
- Default model tags (`qwen3.5:4b`, `nomic-embed-text`) are hard-coded in `AppConfig`, `load_config`, `Profile`, `Profile.from_dict` and `ProfileManager`. Keep them in `AppConfig` only; `load_config` should fall back to the dataclass defaults, and `Profile`/`ProfileManager` should receive them from config (the service already passes them).

## Phase 3 — Frontend excess (no behaviour change)

**3a. One language list.** `constants.js` hard-codes `SUPPORTED_LANGUAGES` while `GET /api/languages` exists and is never called. Fetch `/api/languages` once in `refreshAll()` (alongside health/profiles/app-settings), store it in `state.languages`, build the lookup map from it, and delete the JS copy. Use `native_name` / `english_name` from the API. `index.html` also hard-codes the list three times (composer "Answer in" select ~line 334, default-language select ~line 453, per-language refusal inputs ~line 619). Generate all three from `state.languages` instead, keeping the same element IDs, `data-lang` attributes and option order.

**3b. Translate popover.** `index.html` already contains `#translate-dropdown-popover` with `popover="auto"`, which gives light-dismiss and Escape natively. In `chat.js`:
- Delete the create-if-missing path in `getOrCreateTranslatePopover`, the `fallback-open` / `hidden` fallback branches, and the manual document `click` and `keydown` dismiss listeners. Keep the `toggle` listener (aria-expanded sync) and the resize/scroll close.
- Delete the write-only module variables `activeTranslateTurn`, `activeTranslateProfileId`, `activeTranslateRow`.
- `triggerTranslation(profileId, turn, targetLang, row)` takes the turn object only; remove the index / `row.dataset.turnIdx` lookup paths.
- Remove the duplicate render after `streamSSE` resolves (the `done` handler already sets the cache and renders).
- Positioning: if `positionTranslatePopover` and `infoTips.positionPopover` can share one helper in `utils.js` (flip above/below, clamp horizontally to a container rect), do that; otherwise leave both and say why.

**3c. Info tips.** All 19 `.info-tip-btn` buttons are static in `index.html`, so the label-finding / button-injection block in `initInfoTips()` is dead. Replace it with binding the existing buttons. In `SETTING_INFO_TIPS`, the four `new-profile-*` entries duplicate `setting-*` text — reference the same objects instead of repeating the strings.

**3d. Dead listeners.** `main.js` binds `el.btnSidebarNewChat` and `el.btnHeaderNewChat`, which don't exist in `dom.js` or `index.html`. Delete those two blocks.

**3e. Unnecessary exports.** In `chat.js`, `infoTips.js`, `markdown.js`, `pullModal.js` and `utils.js`, drop `export` from functions only used inside their own module (e.g. `serializeTurnsForSave`, `buildHistoryForRequest`, `positionTranslatePopover`, `triggerTranslation`, `showPopover`, `hidePopover`, `expandCitationExpression`, `normalizeTag`). Check the tests don't import them first.

**3f. CSS.** `.icon-lg` and `.secondary` are not referenced anywhere in HTML/JS. Confirm and delete those rules. No other CSS changes.

After Phase 3, start the app, open http://127.0.0.1:8000, confirm zero console errors, and report the result of each:
1. Ask in English, Malay and Chinese; answers come back in that language with working citations. Debug shows real confidence and rule.
2. "Is the manual available in 中文?" asked in English is answered in English.
3. An off-topic Malay question gets a Malay-only refusal; Debug still shows the score and threshold.
4. Translate an answer; the menu opens, closes on Escape and outside click, and a second translation replaces the first.
5. Info tips open on hover/focus/click and close on Escape in both the settings panel and the New profile dialog.
6. Save chat, reload it, export Markdown (translations included).
7. Change a non-English refusal message and save; the settings history entry mentions it.

## Phase 4 — Tests that check text instead of behaviour
`tests/test_translation.py::test_translation_dropdown_popover_dom_and_css_specs`, `::test_translation_overwrites_previous_translation_in_dom`, and the `styles.css` / `infoTips.js` string-presence tests in `tests/test_info_tooltips.py` assert that strings such as `"z-index: 10000;"` exist in source files. They break on harmless edits and don't test behaviour. Delete them. Keep `test_index_html_contains_all_info_buttons_and_popover` (it checks every tip key has a matching control id) and `test_chunking_and_reindex_wording_accuracy`.

## Out of scope
- Replacing the Markdown renderer, changing the visual design, or new endpoints/features.
- `OllamaBackend._raise_pull_error`'s manifest heuristics — leave as is unless a test shows it misclassifying.

## Final report
Line counts per file before/after, test results, every behaviour change made on purpose (1a–1g, 2e), and anything you noticed but left alone.
