# Refactor Task — Local RAG Chatbot (Version 1)

You are refactoring an existing, working app. Read `local-rag-chatbot-spec.md` and `README.md` first. The goal is to fix a few correctness bugs, remove duplicated code, and split `app/static/app.js` into modules. **User-visible behaviour must not change** except where a phase explicitly says so.

## Ground rules (apply to every phase)

1. **Work one phase at a time.** At the start of each phase, show a short plan (files to touch, what changes). At the end, stop and report: what changed, line counts before/after, test results. Do not start the next phase until I say "continue".
2. **Run the test suite before Phase 0 and after every phase:** `python -m pytest -q`. A phase is only done when all tests pass. If you change behaviour on purpose, update the affected tests and say which ones and why.
3. **Do not touch:** `data/`, `samples/docs/`, `config.yaml`, `requirements.txt`, `styles.css`.
4. **Keep the spec's hard constraints:** the server binds only to 127.0.0.1; it works fully offline; the frontend is plain HTML/CSS/vanilla JS served from `/static` with **no CDN, no npm, no build step**; no new Python dependencies.
5. **Keep public interfaces stable** unless a phase says otherwise: the HTTP API routes and JSON response shapes, `ChatbotService` public methods, and `RAGEngine.query()` (it is used by `tests/` and `samples/run_eval.py`).
6. Prefer small, reviewable edits over rewriting whole files. Don't add features, don't rename things for taste, and don't reformat code you aren't otherwise changing.

---

## Phase 0 — Baseline
- Run `python -m pytest -q` and record the pass/fail count.
- Record the line counts of every file in `app/`.
- Change nothing. Report and stop.

## Phase 1 — Correctness fixes (backend)

**1a. Re-uploading a file can lose the old version.**
In `app/core/service.py` → `ingest_document`, the existing document with the same filename is deleted *before* the new one is chunked, embedded and stored. If embedding or the Chroma write fails, the old document is gone and the new one never gets indexed.
Fix: find the same-name document first, but delete it only **after** the new document has been stored and its chunks added successfully. Add a test: simulate a failing `backend.embed` during a re-upload and assert the original document and its chunks still exist.

**1b. JSON files are overwritten in place.**
`profiles.py` writes `profile.json`, `documents.json` and saved-chat files by opening the target for writing directly. A crash mid-write corrupts the file, and `list_profiles()` then silently skips that profile (`except Exception: continue`), so it vanishes from the UI.
Fix: add one helper, `_atomic_write_json(path, data)`, that writes to a temp file in the same directory and calls `os.replace`. Use it for every JSON write in `profiles.py`. In `list_profiles()`, keep skipping unreadable profiles, but log a warning with the path using the `logging` module instead of passing silently.

**1c. Re-index isn't atomic.**
`_perform_reindex` says it is atomic, but it drops the collection and then re-adds chunks. Keep the current order (embed everything first, then reset and write). Correct the docstring so it doesn't claim atomicity, and add a comment explaining the remaining failure window. No behaviour change.

## Phase 2 — Make the browser the only owner of conversation history

Right now history lives in two places: in server memory (`ChatbotService._histories`, lost on restart) and in the browser's transcript. The browser only sends `history` on retry. On top of that, `GET /api/profiles/{id}/chats/{chat_id}?session_id=...` modifies server state, which a GET shouldn't do.

**Show me the plan and wait for my approval before editing in this phase.** Target design:
- The frontend sends `history` (user/assistant pairs from the active transcript, excluding error turns) with **every** chat request. Use the same filtering as `serializeTurnsForSave` / `get_saved_chat` today.
- The backend uses `body.history` (or an empty list) as the conversation history. Remove `_histories`, `_history_lock`, `get_history`, and the history recording at the end of `stream_chat`.
- `get_saved_chat` no longer takes `session_id` and has no side effects. Keep the query parameter accepted but ignored for one version, so old clients don't get a 422.
- `DELETE /api/profiles/{id}/history`: keep the route as a no-op that returns the same JSON shape, and remove the frontend call to it if there is one.
- `ChatRequest.session_id` stays in the model but is unused.
- Update the tests that depend on server-side session restore (`tests/test_profiles_and_docs.py` around the `session_id="sess-restore"` test, and `tests/test_availability_and_api.py` around `?session_id=api-sess`). Replace them with tests showing that history passed in the request reaches the prompt.

## Phase 3 — Remove backend duplication (no behaviour change)

**3a. `app/core/availability.py`.** The check "find the installed model matching this tag that has capability X" is written out in `evaluate_profile`, `require_ready_for_ingestion` and `require_ready_for_chat`, and copied again in `service._perform_reindex`. The filter `[str(m["name"]) for m in models if "<cap>" in (m.get("capabilities") or [])]` appears about 6 times.
- Add `_names_with_capability(models, cap) -> list[str]` and `_find_model(tag, models, cap) -> dict | None`.
- Add one public method, `resolve_embedding_model(profile) -> str`, that raises `NoModelsInstalledError` or `EmbeddingModelMissingError`. Use it in `require_ready_for_ingestion`, `require_ready_for_chat` and `service._perform_reindex`.
- Put the chat-model + fallback resolution in one private method used by both `evaluate_profile` and `require_ready_for_chat`.

**3b. `app/core/service.py`.**
- Pull the shared parse → chunk → embed steps out of `ingest_document` and `_perform_reindex` into `_prepare_chunks(profile, filename, data, file_hash, embed_model)`. It returns the parse result, the chunks and the embeddings.
- Pull the "estimated chunks" calculation (duplicated in `_serialize_profile` and `update_profile`) into `_count_chunks(profile_id, docs)`.

**3c. Profile fields and defaults live in about 6 places.** These are the `Profile` dataclass defaults, `Profile.from_dict`, `ProfileManager.create_profile`, `ChatbotService.update_profile` (13 fields copied by hand), `ProfilePayload` in `main.py`, and `describe_profile_changes`.
- Make the `Profile` dataclass defaults the single source of truth for default values.
- Add `Profile.with_updates(payload: dict) -> Profile`, built on `dataclasses.replace`, that coerces types and ignores unknown or immutable keys (`id`, `created_at`, `updated_at`, `settings_history`). Use it in `update_profile`, and use the same coercion in `create_profile` and `from_dict`.
- Make `describe_profile_changes` table-driven: a list of `(field, label, formatter)` entries. The output strings must be **identical** to today's, so add a test that pins them.

**3d. `app/core/rag.py`.**
- Pull the three near-identical `meta` event dicts in `stream_query` out into one `_meta_event(...)` builder.
- Pull the history-trimming loop shared by `assemble_rag_messages` and `assemble_conversational_messages` out into `_trim_history(history, max_messages)`.
- Pull the "stream tokens from backend.chat, collect them, yield token events" loop (used twice) out into a helper generator.

**3e. Small items.**
- `ollama.py`: the same httpx exception → domain error mapping is repeated in 5 methods. Handle it in one place, e.g. a context manager `_translate_http_errors(model=None)`. Keep exactly the same error types per method: `pull_model` maps `NetworkError` to `ModelDownloadOfflineError`, and the others map it to `BackendUnavailableError`.
- `Chunk.location_label` and `RetrievedChunk.location_label` are identical. Move the logic into one module-level function and have both properties call it.

## Phase 4 — Split `app/static/app.js` into ES modules (no behaviour change)

`app.js` is 2,676 lines in one closure. Native ES modules need no build step and are allowed by the CSP (`script-src 'self'`).

- Change `index.html` to `<script type="module" src="/static/js/main.js?v=1.2.0"></script>`.
- Proposed layout under `app/static/js/` (adjust if you find better seams, but explain why):
  - `constants.js`: `KNOWN_MODEL_SIZE_ESTIMATES` and `SVG_ICONS`
  - `state.js`: the `state` object
  - `dom.js`: the `el` reference map
  - `utils.js`: escapeHtml, formatBytes, formatTimestamp, formatShortDate, normalizeTag/tagsMatch, clipboard helpers, feedback helpers
  - `api.js`: `apiJson` plus **one new** `streamSSE(url, body, { signal, onEvent })` helper that replaces the copy-pasted fetch → reader → `parseSSEChunkBuffer` loop in both `executeChatQuery` and `executeModelPull`. It must resolve on stream end, reject with the parsed JSON error on a non-OK response, and let AbortError propagate unchanged.
  - `markdown.js`: `formatAssistantMarkdownAndCitations` and `expandCitationExpression`, with output unchanged
  - `banners.js`: `renderProfileBanners`, rewritten around `makeBanner(variant, message, actions)` and `makeModelPicker(models, ariaLabel)` helpers. The four banners must look and behave exactly as they do today.
  - `chat.js`: transcript rendering, streaming, retry, saved chats, Markdown export. The streaming and completed assistant-turn builders share avatar/bubble/refusal-tag scaffolding, so pull that into one function.
  - `settings.js`, `documents.js`, `pullModal.js`: their respective sections
  - `main.js`: `bindEvents()` and startup
- Avoid circular imports. If two modules need each other, pass callbacks or move the shared function down a level.
- Move code first; deduplicate second. Within this phase, do the pure move into modules as one step and confirm the app still works, then add `streamSSE`, the banner helpers and the shared turn scaffolding as a second step.
- There are no frontend tests, so after each step start the app (`python -m app.main`), open http://127.0.0.1:8000 and confirm **zero console errors**, then go through this checklist and report the result of each item:
  1. Page loads; Ollama status pill is correct; sidebar collapse/expand works.
  2. Create a profile, open settings, change and save a setting; the settings history entry appears.
  3. Upload a .md and a .pdf; the document preview, the manager modal, search, the filter pills and Remove all work.
  4. Ask a question that is answered with citations; the citation chips and source pills open the citation modal.
  5. Ask an off-topic question; the refusal tag is shown and the debug panel toggle works.
  6. Send a greeting ("hi"); a conversational reply arrives with no citations.
  7. Stop a stream mid-generation; Retry works on that turn.
  8. Thinking toggle on/off; starter chips work.
  9. Save the chat, start a new chat, reload the saved chat, delete it, and export to Markdown.
  10. Open the pull-model modal; the confirmation step shows the size estimate; Cancel works (no actual download needed).
  11. Switch the embedding model in settings and the re-index confirmation dialog appears; Cancel leaves the setting unchanged.

## Out of scope (do not do)
- Replacing the hand-written markdown renderer with a library.
- Any change to `styles.css` or to the visual design.
- New features, API endpoints, or dependencies.

## Final report (after Phase 4)
Give a table of line counts per file before and after, the test results, any behaviour you changed intentionally, and anything you noticed but deliberately left alone.
