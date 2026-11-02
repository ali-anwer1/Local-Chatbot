# Project Prompts & Implementation History

This directory archives all prompt specifications and instructions executed in chronological order:

| Order | Prompt File | Scope & Purpose |
| :---: | :--- | :--- |
| **01** | [`01_initial_spec_phase1_local_rag.md`](01_initial_spec_phase1_local_rag.md) | **Initial Project Specification (Phase 1):** Built the foundational 100% offline local RAG chatbot architecture with multi-profile isolation, ChromaDB vector storage, PDF/DOCX/MD/TXT parsers, code-enforced pre-LLM similarity refusal gate, and Ollama integration. |
| **02** | [`02_refactor_backend_fixes_client_history_modules.md`](02_refactor_backend_fixes_client_history_modules.md) | **Refactor & Architecture Hardening:** Fixed backend re-upload safety and atomic JSON writes (`_atomic_write_json`), transferred conversation history ownership exclusively to the browser client, deduplicated model capability checks in `availability.py`, and refactored monolithic `app.js` into modular ES modules. |
| **03** | [`03_pre_phase2_cpu_embedding_ui_redesign_setting_tooltips.md`](03_pre_phase2_cpu_embedding_ui_redesign_setting_tooltips.md) | **Hardware Optimization, UI Redesign & Info Tooltips:** Added switchable CPU/Auto embedding execution (`options.num_gpu = 0`) to preserve GPU/iGPU memory for chat models, configured `keep_alive`, added detailed latency/throughput metrics in Debug mode, redesigned the UI into a purpose-built internal tool with local IBM Plex typography, and created accessible setting info tooltips (`infoTips.js`). |
| **04** | [`04_multilingual_answers_and_on_demand_translation.md`](04_multilingual_answers_and_on_demand_translation.md) | **Multilingual Intelligence & Answer Translation:** Added 8-language support with offline Lingua detection, native refusal messages, and composer selector; built on-demand faithful answer translation (`POST /api/translate`) with clean in-place overwriting, in-memory caching, and top-layer Popover positioning. |
