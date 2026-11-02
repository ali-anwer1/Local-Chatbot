# Pre-Phase-2 changes: embedding device, UI redesign, settings info tooltips

Make the three changes below to this codebase. Read the existing code first (`app/core/backends/ollama.py`, `app/core/rag.py`, `app/core/service.py`, `app/config.py`, `app/static/index.html`, `app/static/styles.css`, `app/static/js/*`). Keep the existing architecture: FastAPI + vanilla JS ES modules, no frontend framework, no build step. The app must stay fully offline: no CDNs, no remote fonts, no external requests. Keep all existing tests passing and add tests where noted. Do the three parts in order and stop after each to summarise what changed.

---

## Part 1 — Run the embedding model on CPU (switchable), and measure speed

Goal: keep the iGPU's memory free for the chat model by running the embedding model (default `nomic-embed-text`) on CPU, with a switch back to letting Ollama decide.

### Backend
1. Add an app-wide setting `embedding_device` with values `"cpu"` or `"auto"`. Default `"cpu"`. This is a machine-level setting, not a per-profile one.
   - Default comes from `config.yaml` (add the key with a comment).
   - Runtime changes from the UI persist to `data/app_settings.json`, which overrides `config.yaml` values on load. `AppConfig` is frozen, so add a small mutable app-settings holder rather than mutating it.
   - Add `GET /api/app-settings` and `PUT /api/app-settings` (validate values; reject anything other than `cpu`/`auto`).
2. In `OllamaBackend.embed()`, when `embedding_device == "cpu"`, send `"options": {"num_gpu": 0}` in every `/api/embed` request payload (both ingestion and query-time embedding). When `"auto"`, omit `num_gpu`. Always send the same options for a given setting so Ollama does not reload the model between calls.
3. Add an app-wide `keep_alive` setting (default `"30m"`, in `config.yaml`) and send it on both `/api/chat` and `/api/embed` requests so neither model is unloaded between questions.
4. Changing the device does NOT require re-indexing: vectors are the same model's output. Do not trigger a re-index.

### Timing instrumentation (Debug mode only)
5. Record per-response timings and show them in the existing Debug view under each answer:
   - `embed_ms` (query embedding), `retrieval_ms` (Chroma query), `time_to_first_token_ms`, `total_ms`
   - From Ollama's final stream chunk (`done: true`): `prompt_eval_count`, `prompt_eval_duration`, `eval_count`, `eval_duration` → show prompt tokens, prompt processing time, generated tokens, and tokens/second.
   - `OllamaBackend.chat()` currently yields only strings and discards the final chunk's stats. Capture them without breaking the current interface (e.g. store the last call's stats on the backend instance, or emit a separate stats event that `rag.stream_query` forwards in its existing `done` event). Update the `LLMBackend` base class if needed.

### UI
6. In the Settings panel, add a section titled "Performance (applies to all profiles)" containing a select: "Embedding device" with options "CPU" and "Auto (Ollama decides)". Save via the new endpoint. Show a one-line note: "Takes effect on the next request; the embedding model reloads once."

### Tests
7. Add tests: embed payload contains `options.num_gpu == 0` when device is `cpu` and has no `num_gpu` when `auto`; `keep_alive` present on chat and embed payloads; app-settings endpoint validation and persistence; timing fields present in the `done` event when available.

---

## Part 2 — Redesign the frontend so it doesn't look like a generic AI-generated app

The current UI has the common tells of template AI apps: saturated blue accent on navy slate, a gradient icon tile with a star logo, a centered empty-state hero ("Ask X anything" + tagline + pill-shaped suggestion chips), pill badges everywhere, large uniform rounded corners, outlined buttons lined up in the header, soft shadows on everything, Title Case labels, and marketing copy ("Private Knowledge AI", "Grounded strictly in…"). Replace this with the look of a calm, purpose-built internal tool: neutral, dense, typographic, few decorations.

### Design tokens (replace the palette in `styles.css`; keep the existing light/dark `prefers-color-scheme` mechanism and CSS layers)
- **Dark:** app bg `#141414`, sidebar `#1a1a1a`, surface `#1f1f1f`, hover `#262626`, borders `#2a2a2a` / strong `#3a3a3a`, text `#e8e6e3`, secondary `#bdb9b3`, muted `#8f8b85`.
- **Light:** app bg `#fafaf8`, sidebar `#f3f2ef`, surface `#ffffff`, hover `#ebe9e5`, borders `#e3e1dc` / strong `#cfccc6`, text `#1c1b19`, secondary `#45423d`, muted `#77736c`.
- **One accent only**, used sparingly (focus rings, active-profile marker, send button, links): ochre — dark `#c9a25a`, light `#8a6418`. No other blues, no gradients, no glows.
- Status colours (success/warning/danger) stay but desaturate them to match.
- **Fonts:** IBM Plex Sans (UI and body) and IBM Plex Mono (numbers, model tags, chunk IDs, code). Bundle the `.woff2` files locally under `app/static/fonts/` (e.g. take them from the `@fontsource/ibm-plex-sans` and `@fontsource/ibm-plex-mono` npm packages, weights 400/500/600) and load with `@font-face`. Fall back to the system stack if missing.
- **Type scale:** 13px UI chrome, 15px chat body with line-height 1.6, 12px meta text. Use `font-variant-numeric: tabular-nums` for counts and scores.
- **Shape:** border-radius 6px for inputs/buttons, 8px for panels/modals. No pill shapes anywhere except the small status dot.
- **Depth:** no shadows except on modals and popovers. Separate regions with 1px borders and spacing, not cards-in-cards.
- **Motion:** colour/background transitions only, 120ms. No hover lift, no scale, no shimmer.

### Specific changes
1. **Brand block:** remove the gradient star tile and the "Private Knowledge AI" tagline. Use a plain text wordmark "Local RAG" (600 weight, 15px).
2. **Sidebar:** profiles as plain list rows (name + muted "8 docs · 82 chunks" in mono). Active profile = slightly raised background + 2px accent bar on the left edge, not a filled blue card. Remove the sidebar's "New Chat" button OR the header's one — there are currently two; keep only the header one.
3. **Ollama status (sidebar footer):** small coloured dot + muted text "Ollama online · 11 models", no filled green pill. "Download model…" becomes a text-style button.
4. **Chat header:** profile name, then metadata as plain muted text ("qwen3.5:4b · 8 docs · 82 chunks"), not pill badges. Right side: "New chat" as the only visible text button; Save chat, Export and Debug move into a single "⋯" overflow menu (keyboard accessible, closes on Escape/outside click). Settings stays as an icon button.
5. **Empty state:** remove the centered icon, "Ask … anything" headline, tagline and pill chips. Replace with a left-aligned block at the top of the chat column showing: the profile name, the first line of its preamble (muted), a compact list of the documents in its knowledge base (filename + type, max 8, then "+N more"), and 2–3 example questions as plain text links that fill the composer when clicked.
6. **Messages:** assistant answers render as plain full-width text (no bubble), max column width ~720px. User messages: a subtle surface block with a small "You" label, no rounded bubble. Citations render as small superscript numbers `[1]` that link to a compact "Sources" list under the answer (filename + chunk number in mono).
7. **Composer:** a single bordered box anchored to the bottom of the chat column, 8px radius, no floating shadow. Thinking toggle becomes a small text toggle ("Thinking off/on"). Send button: 32px square, 6px radius, accent background. Keyboard hint in 12px muted text.
8. **Settings panel:** group fields under small section headings (Profile, Models, Retrieval, Indexing, Conversation, Performance). Labels above inputs, consistent field widths, helper text 12px muted.
9. **Copy:** sentence case everywhere ("Save chat", "New chat", "Bot profiles" → "Profiles"). Remove marketing phrasing. Use "…" (single character) for truncation.
10. **Icons:** one consistent inline-SVG set (Lucide-style outlines, 1.5px stroke, 16px). No emoji, no star logo.

### Constraints
- Keep every element ID and `aria-*` attribute the JS depends on; if you must rename or restructure, update the JS in the same change.
- Keep all existing functionality and both colour schemes working.
- Check contrast: body text and muted text must meet WCAG AA against their backgrounds in both themes.

---

## Part 3 — Info icons on every setting

Add a small info icon (ⓘ-style outline SVG, 14px, muted, accent on hover/focus) after each setting label in the Settings panel and in the New Profile modal.

### Behaviour
- Hover or keyboard focus shows a popover; click/tap toggles it (pins it open on touch). Escape or outside click closes it. Only one popover open at a time.
- The icon is a `<button type="button">` with `aria-label="About <setting name>"` and `aria-describedby` pointing at the popover; popover has `role="tooltip"`.
- Popover: max-width 280px, 13px text, surface background, 1px border, 6px radius, small shadow; positions itself to stay inside the settings panel (flip above/below as needed).
- Build it as one reusable module (`app/static/js/infoTips.js`) and keep all text in one object (e.g. in `constants.js`) keyed by setting ID, so wording can be edited in one place.
- Remove the inline `field-hint` text that duplicates a tooltip, but keep hints that warn about consequences (e.g. the embedding-model re-index warning).

### Text (verify each against the actual code behaviour and correct the wording if the code differs — in particular whether changing chunk size/overlap re-indexes existing documents or only affects new uploads)
- **Display name:** Name shown in the sidebar and chat header. Cosmetic only.
- **System preamble:** Instructions sent to the model before every conversation: the bot's role, tone and scope. It does not add knowledge — facts must come from the uploaded documents.
- **Chat model:** The model that writes the answers. Larger models usually answer better but are slower and use more memory.
- **Thinking:** Lets models that support it reason step by step before answering. Can help on complex questions but makes responses much slower. No effect on models without thinking support.
- **Embedding model:** Turns documents and questions into vectors so relevant passages can be found. Documents and questions must use the same embedding model, so changing it re-indexes every document.
- **Embedding device:** Where the embedding model runs. CPU leaves the iGPU's memory for the chat model; Auto lets Ollama decide. Applies to all profiles.
- **Refusal similarity threshold:** How closely the best-matching passage must match the question (0–1) before the model is asked to answer. Below this, the bot replies with the refusal message without calling the model. Higher = stricter: fewer unsupported answers, more refusals. Lower = answers more often, with more risk of off-topic answers.
- **Temperature:** How much the wording varies. Low (0–0.3) keeps answers consistent and close to the sources; higher values are more varied and more likely to drift.
- **Top-K chunks:** How many of the best-matching passages are given to the model with each question. More covers more ground but makes the prompt longer and responses slower.
- **Chunk size:** Length, in characters, of the pieces documents are split into. Smaller pieces match questions more precisely but carry less context each; larger pieces carry more context but match less precisely.
- **Overlap:** Characters repeated between neighbouring chunks so sentences at a boundary aren't cut in half. Usually 10–20% of the chunk size.
- **Conversation history turns:** How many previous question-and-answer pairs are sent with each new question, so follow-ups like "what about the second one?" work. More turns = better follow-ups but longer prompts and slower responses.
- **Refusal message:** The exact text shown when the answer can't be found in the knowledge base.

---

## Done when
- All tests pass (`pytest`), including the new ones.
- With Debug on, each answer shows the timing breakdown.
- The UI matches Part 2 in both light and dark mode, with no external network requests (check the browser's Network tab).
- Every setting has a working info icon, usable by mouse, keyboard and touch.
