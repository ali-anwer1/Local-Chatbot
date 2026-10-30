# Local RAG Chatbot

A 100% local, offline-capable Retrieval-Augmented Generation (RAG) chatbot designed for grounded Q&A over user-supplied document collections.

Each **bot profile** pairs a custom system preamble with its own isolated persistent vector collection (`PDF`, `DOCX`, `MD`, and `TXT`), backed by local [Ollama](https://ollama.com) completion and embedding models.

## Key Features

- **Strict Grounding, Conversational Awareness & Code-Enforced Refusal Gate:** Natural greetings and capability overview queries (*"What can you help me with?"*) respond warmly using the profile's persona and uploaded document list, while factual retrieval similarity is checked in Python *before* calling the LLM—if no passage meets `similarity_threshold`, the bot refuses immediately without calling the model.
- **Cited-Only Verifiable Sources:** Answers cite retrieved passages inline (`[1]`, `[2]`), and the response footer displays **only** the passages actually used in the answer (while the optional **Debug** view inspects all `top_k` candidate chunks and scores).
- **Fast Responses, Live Progress & Thinking Toggle:** Model thinking mode is disabled by default (`think: false`) for snappy responses, with a one-click **Thinking** toggle in the composer and live stage progress indicators during retrieval and generation.
- **Saved Chats, Export & Organised Document Manager:** Save conversations per profile in the collapsible left sidebar (or export to `.md`), copy previous queries, retry failed turns, and browse/search large knowledge bases via a dedicated Document Manager modal.
- **Multi-Profile Isolation:** Create separate profiles (e.g., *Astra NX-200 Hardware Assistant*, *Engineering SOP Q&A*), each with its own preamble, model choices, threshold, settings change history, and isolated ChromaDB collection.
- **100% Offline Operation:** Binds strictly to `127.0.0.1`, disables Chroma telemetry, and serves a modern Gemini/ChatGPT-style HTML/CSS/JS UI with zero external CDNs.
- **Resilient Model Availability (§4.7):** Handles offline Ollama, missing models, explicit chat fallbacks, embedding model mismatch protection, confirmed re-indexing, and out-of-memory load errors gracefully.

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
   *(After completing setup, you can disconnect from Wi-Fi/Ethernet—everything runs offline on `127.0.0.1`.)*

---

## Demo Knowledge Base & Evaluation (`samples/`)

The `samples/` directory ships with a portfolio-safe enterprise hardware knowledge base for the **Astra NX-200 Industrial Edge Gateway**:

- `samples/docs/astra_nx200_datasheet.pdf` — 3-page hardware datasheet (compute, electrical, environmental, fieldbus I/O)
- `samples/docs/astra_nx200_maintenance_sop.docx` — Word SOP with maintenance schedule table, thermal pad torque, and firmware recovery steps
- `samples/docs/astra_warranty_and_sla.md` — Warranty coverage, exclusions, and P1–P3 RMA SLAs
- `samples/docs/astra_field_faq.txt` — Admin login, LED fault codes (`E-409`), and DIN-rail airflow clearances

### Try the Browser Demo
1. Click the **`+`** icon next to **Bot Profiles** in the left sidebar and create e.g. `Astra NX-200 Spec Assistant`.
2. Click the **Settings & Knowledge Base sliders icon** in the top-right header, click **Upload Files…**, and select all 4 files in `samples/docs/` (use **Manage Files** to search/filter all uploaded documents).
3. Try clicking **"What can you help me with?"** or asking a technical question, toggle **Debug** in the top toolbar to inspect live candidate chunks and scores, and click **Save Chat** or **Export** to save your conversation.

### Run the Automated Evaluation Script
With Ollama running locally and `qwen3.5:4b` + `nomic-embed-text` installed:

```powershell
python samples/run_eval.py
```

Optional threshold calibration override:

```powershell
python samples/run_eval.py --threshold 0.50
```

The script ingests all 4 sample documents into a temporary profile, runs 10 in-scope questions and 5 out-of-scope questions (`samples/eval_questions.json`), and reports overall pass/fail, citation accuracy, and refusal accuracy.

---

## Running Unit Tests

All unit and API tests mock the LLM backend and run completely offline without needing Ollama running:

```powershell
pytest -v
```

---

## Configuration (`config.yaml`)

```yaml
host: "127.0.0.1"              # Locked to loopback only
port: 8000
data_dir: "data"               # Profiles, uploaded docs, and ChromaDB (gitignored)

ollama_base_url: "http://127.0.0.1:11434"
default_chat_model: "qwen3.5:4b"
default_embedding_model: "nomic-embed-text"
allow_chat_fallback: true
model_cache_ttl_seconds: 30
```

---

## Project Structure

```text
├── app/
│   ├── main.py              # Thin FastAPI app & HTTP/SSE routes only
│   ├── config.py            # Loads and validates config.yaml
│   ├── core/                # Framework-agnostic business logic
│   │   ├── availability.py  # Model cache, fallback rules, metadata checks
│   │   ├── errors.py        # Structured user-safe domain exceptions
│   │   ├── ingestion.py     # PDF/DOCX/MD/TXT parsers, scanned check & chunker
│   │   ├── profiles.py      # Profile & document manifest persistence
│   │   ├── rag.py           # Query builder, refusal gate & prompt assembly
│   │   ├── service.py       # Domain orchestrator & session memory
│   │   ├── vectorstore.py   # Isolated persistent ChromaDB collections
│   │   └── backends/
│   │       ├── base.py      # Abstract LLMBackend interface
│   │       └── ollama.py    # Ollama REST API implementation
│   └── static/              # Offline HTML, modern CSS, and vanilla JS UI
├── data/                    # Runtime profiles, docs & ChromaDB (gitignored)
├── samples/                 # Demo mixed-format documents & evaluation runner
├── tests/                   # Unit & integration test suite
├── config.yaml
├── requirements.txt
└── README.md
```
