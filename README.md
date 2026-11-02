# Local RAG Chatbot

A 100% local, offline-capable Retrieval-Augmented Generation (RAG) chatbot designed for grounded Q&A over user-supplied document collections.

Each **bot profile** pairs a custom system preamble with its own isolated persistent vector collection (`PDF`, `DOCX`, `MD`, and `TXT`), backed by local [Ollama](https://ollama.com) completion and embedding models.

---

## Key Features

- **Strict Grounding, Conversational Awareness & Code-Enforced Refusal Gate:** Natural greetings and capability queries (*"What can you help me with?"*) respond warmly using the profile's persona and uploaded document list, while factual retrieval similarity is checked in Python *before* calling the LLM—if no passage meets `similarity_threshold`, the bot refuses immediately in code without invoking the model.
- **Native Multilingual Intelligence (8 Languages):** Understands and responds in English, Malay (Bahasa Melayu), Simplified Chinese (简体中文), Tamil (தமிழ்), Indonesian (Bahasa Indonesia), Japanese (日本語), Korean (한국어), and Arabic (العربية). Fast statistical detection via Lingua runs locally, complemented by explicit regex patterns and a composer-level language override.
- **On-Demand Faithful Answer Translation:** Translate any assistant response into any of the 8 supported languages on-demand. Translation runs with temperature 0.1 and `think: false`, strictly preserving citation markers (`[n]`), numbers, units, product models, and specifications. Overwrites the translation in-place to keep the chat uncrowded, caches results in memory for instant restoration, and uses a top-layer Popover that never clips through message rows or gets obscured by the composer.
- **Switchable Embedding Device (CPU / Auto) & Performance Metrics:** Run the embedding model on CPU (`embedding_device: "cpu"`) to keep iGPU/GPU VRAM fully available for the chat model, or switch to Auto. High-precision Debug mode inspects query embedding latency, Chroma retrieval time, time to first token (TTFT), total response time, and generated tokens/sec.
- **Verifiable Inline Citations & Filtered Sources:** Answers cite retrieved passages inline (`[1]`, `[2]`), and the response footer displays **only** the passages actually cited in the answer (while the optional **Debug** inspector reveals all `top_k` candidate chunks and scores). Click any citation chip to view exact chunk text and document location.
- **Purpose-Built Internal-Tool UI & Offline Typography:** Designed as a calm, dense, neutral tool: IBM Plex Sans and IBM Plex Mono fonts bundled locally (zero remote CDN dependencies), single ochre accent (`#c9a25a` dark, `#8a6418` light), tabular numbers, flat 1px borders, and sentence-case typography.
- **Accessible Setting Info Tooltips:** Every configuration parameter in the Settings panel and New Profile modal features an accessible `ⓘ` info popover explaining its behavior, tradeoffs, and indexing implications (via `infoTips.js`).
- **Saved Chats, Markdown Export & Document Manager:** Save and resume conversation sessions per profile in the collapsible left sidebar, export chats to clean `.md` with full translation history, copy previous turns, retry queries, and search/filter large document sets in the dedicated Document Manager modal.
- **Multi-Profile Isolation:** Create independent profiles (e.g. *Astra NX-200 Hardware Assistant*, *Engineering SOP Q&A*), each with its own preamble, models, threshold, settings history, and isolated ChromaDB collection.
- **100% Offline & Private:** Binds strictly to loopback `127.0.0.1`, disables Chroma telemetry, and loads all assets locally with zero internet access required.
- **Resilient Model Availability (§4.7):** Gracefully handles offline Ollama, missing models, explicit chat fallbacks, embedding model mismatch detection, confirmed re-indexing workflows, and out-of-memory (OOM) load errors.

---

## Supported Languages & Precedence

The chatbot supports 8 languages across language detection, prompt engineering, native refusal messages, and on-demand answer translation:

| Code | English Name | Native Name | Instruction Name |
| :--- | :--- | :--- | :--- |
| `en` | English | English | English |
| `ms` | Malay | Bahasa Melayu | Bahasa Melayu (Malay) |
| `zh` | Simplified Chinese | 简体中文 | Simplified Chinese |
| `ta` | Tamil | தமிழ் | Tamil |
| `id` | Indonesian | Bahasa Indonesia | Bahasa Indonesia |
| `ja` | Japanese | 日本語 | Japanese |
| `ko` | Korean | 한국어 | Korean |
| `ar` | Arabic | العربية | Arabic |

### Language Selection Precedence
When answering a query, the target language is resolved in the following strict order:
1. **Composer Selector:** An explicit per-chat "Answer in: [Language]" selection in the composer.
2. **In-Text Instruction:** Explicit requests in the user prompt (e.g. *"answer in Malay"*, *"用中文回答"*, *"jawab dalam bahasa melayu"*).
3. **Statistical Detection:** Offline detection of the latest user message via `lingua-language-detector` (with an explicit Malay-over-Indonesian tie-break for regional context).
4. **Conversation Fallback:** The language of the preceding user turn in the active chat.
5. **Profile Default:** The profile's configured `default_language` (defaults to English).

---

## Quick Setup (7 Steps)

1. **Install Python 3.11+** and **[Ollama](https://ollama.com/download)**.
2. **Start Ollama** and pull the default chat and embedding models:
   ```powershell
   ollama pull qwen3.5:4b
   ollama pull nomic-embed-text
   ```
3. **Open a terminal** in the project root directory.
4. **Create a Python virtual environment:**
   - Windows (PowerShell):
     ```powershell
     python -m venv .venv
     ```
   - Linux / macOS:
     ```bash
     python3 -m venv .venv
     ```
5. **Activate the virtual environment:**
   - Windows (PowerShell):
     ```powershell
     .\.venv\Scripts\Activate.ps1
     ```
   - Linux / macOS:
     ```bash
     source .venv/bin/activate
     ```
6. **Install Python dependencies:**
   ```powershell
   pip install -r requirements.txt
   ```
7. **Start the local server** and open **http://127.0.0.1:8000** in your browser:
   ```powershell
   python -m app.main
   ```
   *(After setup, you can disconnect from Wi-Fi or Ethernet—everything runs 100% offline on `127.0.0.1`.)*

---

## Hardware & Performance Controls

### Switchable Embedding Device (CPU / Auto)
By default, the embedding model runs on CPU (`options: {"num_gpu": 0}` sent to Ollama's `/api/embed`). This frees integrated GPU (iGPU) and discrete GPU memory exclusively for the chat generation model, preventing out-of-memory errors on modest hardware.
- Can be toggled between **CPU** and **Auto (Ollama decides)** in the Settings panel under **Performance**.
- Does **not** require re-indexing, as embeddings generated by the same model on CPU or GPU are mathematically identical.
- Machine-level setting stored in `config.yaml` and persisted across runtime changes in `data/app_settings.json`.

### Keep-Alive
An app-wide `keep_alive: "30m"` configuration prevents Ollama from unloading the chat and embedding models between questions, eliminating cold-start latency during active sessions.

### Debug Performance Telemetry
When the **Debug** toggle is enabled in the top toolbar, every assistant answer reveals an exact latency and throughput breakdown:
- **Query Embedding (`embed_ms`):** Time spent converting the query into an embedding vector.
- **Vector Retrieval (`retrieval_ms`):** ChromaDB vector similarity query time.
- **Time to First Token (`time_to_first_token_ms`):** Initial LLM generation response latency.
- **Generation Throughput (`tokens/sec`):** Generation speed calculated from Ollama's eval metrics (`eval_count` and `eval_duration`).
- **Token Counts:** Evaluated prompt tokens (`prompt_eval_count`) and generated response tokens (`eval_count`).

---

## Demo Knowledge Base & Multilingual Evaluation (`samples/`)

The `samples/` directory includes a portfolio-safe enterprise hardware knowledge base for the **Astra NX-200 Industrial Edge Gateway**:

- `samples/docs/astra_nx200_datasheet.pdf` — Hardware datasheet (compute, electrical, environmental, fieldbus I/O)
- `samples/docs/astra_nx200_maintenance_sop.docx` — SOP with maintenance schedule table, thermal pad torque, and firmware recovery
- `samples/docs/astra_warranty_and_sla.md` — Warranty coverage, exclusions, and RMA SLAs
- `samples/docs/astra_field_faq.txt` — Admin credentials, LED fault codes (`E-409`), and DIN-rail airflow clearances

### Try the Browser Demo
1. Click the **`+`** icon next to **Profiles** in the left sidebar to create e.g. `Astra NX-200 Spec Assistant`.
2. Open the **Settings & Knowledge Base** drawer (top-right sliders icon), click **Upload Files…**, and select all 4 files in `samples/docs/`.
3. Ask a technical question, test conversational queries, toggle the **Thinking** mode or composer **Answer in** language selector, and click **Translate ▾** under any answer to translate into Malay, Mandarin, or other languages.

### Run the Automated Evaluation Scripts
With Ollama running locally and `qwen3.5:4b` + `nomic-embed-text` installed:

- **Standard English evaluation:**
  ```powershell
  python samples/run_eval.py
  ```
- **Malay evaluation:**
  ```powershell
  python samples/run_eval.py --questions samples/eval_questions_ms.json
  ```
- **Simplified Chinese evaluation:**
  ```powershell
  python samples/run_eval.py --questions samples/eval_questions_zh.json
  ```
- **Complete multilingual comparison suite:**
  ```powershell
  python samples/run_eval.py --all
  ```
- **Test with a multilingual embedding model (e.g. `bge-m3`):**
  ```powershell
  python samples/run_eval.py --embedding-model bge-m3 --all
  ```

The evaluation runner ingests all sample documents into an isolated temporary profile, executes in-scope and out-of-scope test questions, and outputs pass rates, citation accuracy, mean retrieval scores, wrong refusals, and language fidelity.

---

## Running Unit Tests

All unit, API, multilingual, and translation tests mock the LLM backend and run completely offline without needing an active Ollama instance:

```powershell
pytest -v
```

All 57 unit tests pass across:
- Model availability and fallback states (`test_availability_and_api.py`)
- Ingestion, chunking, and scanned PDF detection (`test_ingestion.py`)
- RAG query generation, similarity refusal gate, and live streaming (`test_rag_and_gate.py`)
- Profile CRUD, document deduplication, and persistence (`test_profiles_and_docs.py`)
- Language detection, override precedence, and native refusals (`test_multilingual.py`)
- On-demand translation prompt, streaming SSE, Popover DOM specs, and overwrite behavior (`test_translation.py`)
- Setting info tooltips constants and popover assertions (`test_info_tooltips.py`)

---

## Configuration (`config.yaml` & `data/app_settings.json`)

```yaml
host: "127.0.0.1"              # Locked strictly to loopback
port: 8000
data_dir: "data"               # Profiles, uploaded docs, and ChromaDB (gitignored)

ollama_base_url: "http://127.0.0.1:11434"
default_chat_model: "qwen3.5:4b"
default_embedding_model: "nomic-embed-text"
allow_chat_fallback: true
model_cache_ttl_seconds: 30

# Device for the embedding model: "cpu" (preserves GPU/iGPU memory for chat) or "auto"
embedding_device: "cpu"

# Memory retention window for Ollama models between queries
keep_alive: "30m"
```

Runtime changes to `embedding_device` made via the UI or API are stored in `data/app_settings.json`, safely overriding `config.yaml` without mutating configuration files.

---

## API Reference

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `POST` | `/api/chat` | Stream RAG or conversational response via SSE (`status`, `token`, `done`, `error`). Supports `requested_language`. |
| `POST` | `/api/translate` | Stream faithful answer translation via SSE (`status`, `token`, `done`, `error`) using chat model without retrieval. |
| `GET` | `/api/languages` | Return list of supported languages (codes, English names, native names). |
| `GET` | `/api/app-settings` | Retrieve global application settings (`embedding_device`, `keep_alive`). |
| `PUT` | `/api/app-settings` | Update global application settings (`embedding_device`). |
| `GET` | `/api/profiles` | List all bot profiles and availability status. |
| `POST` | `/api/profiles` | Create a new bot profile. |
| `GET` | `/api/profiles/{id}` | Get full profile metadata, settings history, and document manifest. |
| `PUT` | `/api/profiles/{id}` | Update profile configuration (preamble, models, threshold, languages, refusals). |
| `DELETE` | `/api/profiles/{id}` | Delete profile and remove its persistent ChromaDB collection. |
| `POST` | `/api/profiles/{id}/documents` | Upload and ingest documents (`.pdf`, `.docx`, `.md`, `.txt`). |
| `DELETE` | `/api/profiles/{id}/documents/{doc_id}` | Delete document and prune its chunk vectors from ChromaDB. |
| `POST` | `/api/profiles/{id}/reindex` | Re-index all profile documents (required when changing embedding models). |
| `GET` | `/api/profiles/{id}/saved-chats` | List saved conversation sessions for the profile. |
| `POST` | `/api/profiles/{id}/saved-chats` | Save a conversation session (includes translations and citation links). |
| `DELETE` | `/api/profiles/{id}/saved-chats/{chat_id}` | Delete a saved conversation session. |
| `GET` | `/api/ollama/models` | List installed Ollama models and capability flags. |

---

## Project Structure

```text
├── app/
│   ├── main.py              # Thin FastAPI application & HTTP/SSE route handlers
│   ├── config.py            # Loads and validates config.yaml & app_settings.json
│   ├── core/                # Framework-agnostic business logic
│   │   ├── availability.py  # Model capability checks, fallback rules & cache
│   │   ├── errors.py        # Structured domain exceptions
│   │   ├── ingestion.py     # PDF/DOCX/MD/TXT parsers, scanned PDF check & chunker
│   │   ├── languages.py     # 8 languages, Lingua detector, regex patterns, translation prompt
│   │   ├── profiles.py      # Profile manifest persistence & change history logging
│   │   ├── rag.py           # Similarity refusal gate, grounding & prompt assembler
│   │   ├── service.py       # Domain orchestrator, session memory & translation
│   │   ├── vectorstore.py   # Isolated persistent ChromaDB collections
│   │   └── backends/
│   │       ├── base.py      # Abstract LLMBackend interface & timing stats
│   │       └── ollama.py    # Ollama REST API implementation (CPU/GPU options)
│   └── static/              # 100% offline UI (zero CDNs)
│       ├── fonts/           # Bundled IBM Plex Sans and IBM Plex Mono (.woff2)
│       ├── js/              # Modular vanilla JS modules
│       │   ├── api.js       # HTTP fetch and SSE stream parser
│       │   ├── banners.js   # Ollama connection & status banners
│       │   ├── chat.js      # Transcript rendering, streaming, translation popover
│       │   ├── constants.js # Languages, setting tooltips content & SVG icons
│       │   ├── documents.js # Document manager modal & upload handlers
│       │   ├── dom.js       # Cached DOM element references
│       │   ├── infoTips.js  # Accessible setting info popover controller
│       │   ├── main.js      # App initialization & tab wiring
│       │   ├── markdown.js  # Fast local markdown & citation formatter
│       │   ├── pullModal.js # Offline model downloader modal
│       │   ├── settings.js  # Settings drawer & history log UI
│       │   ├── state.js     # Reactive client-side application state
│       │   └── utils.js     # Clipboard, escaping, and formatting helpers
│       ├── index.html       # Semantic HTML layout with Popover API elements
│       └── styles.css       # Clean typographic internal-tool CSS
├── data/                    # Local runtime storage (profiles, docs, vectors - gitignored)
├── prompts/                 # Project prompts & specifications archive
│   ├── README.md            # Chronological prompt index & changelog summary
│   ├── 01_initial_spec_phase1_local_rag.md
│   ├── 02_refactor_backend_fixes_client_history_modules.md
│   ├── 03_pre_phase2_cpu_embedding_ui_redesign_setting_tooltips.md
│   └── 04_multilingual_answers_and_on_demand_translation.md
├── samples/                 # Enterprise hardware evaluation suite
│   ├── docs/                # Astra NX-200 PDF, DOCX, MD, and TXT knowledge base
│   ├── eval_questions.json  # English benchmark questions
│   ├── eval_questions_ms.json # Malay benchmark questions
│   ├── eval_questions_zh.json # Simplified Chinese benchmark questions
│   ├── generate_samples.py  # Sample documents generator
│   └── run_eval.py          # Multilingual benchmark evaluation runner
├── tests/                   # 57 passing offline unit & integration tests
│   ├── conftest.py          # Deterministic mock backend & fixtures
│   ├── test_availability_and_api.py
│   ├── test_info_tooltips.py
│   ├── test_ingestion.py
│   ├── test_multilingual.py
│   ├── test_profiles_and_docs.py
│   ├── test_rag_and_gate.py
│   └── test_translation.py
├── config.yaml              # Server & default model configuration
├── requirements.txt         # Pinned Python package dependencies
└── README.md
```
