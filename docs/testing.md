[← README](../README.md) · [Docs](README.md) · [Architecture](architecture.md) · [Configuration](configuration.md) · [Languages](languages.md) · [API reference](api.md) · **Testing**

# Testing and evaluation

## Unit tests

```powershell
pytest -v
```

The suite uses a mocked LLM backend (see [`conftest.py`](../tests/conftest.py)), so it runs offline and doesn't need Ollama.

| File | Covers |
| :--- | :--- |
| [`test_availability_and_api.py`](../tests/test_availability_and_api.py) | Ollama/model checks, fallbacks, API responses |
| [`test_ingestion.py`](../tests/test_ingestion.py) | Parsing, chunking, scanned-PDF detection |
| [`test_rag_and_gate.py`](../tests/test_rag_and_gate.py) | Retrieval, refusal gate, streaming |
| [`test_profiles_and_docs.py`](../tests/test_profiles_and_docs.py) | Profile and document storage, duplicates |
| [`test_multilingual.py`](../tests/test_multilingual.py) | Language detection and precedence, localized refusals |
| [`test_translation.py`](../tests/test_translation.py) | Translation prompt, streaming and behaviour |
| [`test_info_tooltips.py`](../tests/test_info_tooltips.py) | Settings tooltip content |

## Evaluation runner

[`samples/run_eval.py`](../samples/run_eval.py) measures answer quality against the [Astra NX-200 sample documents](../samples/docs/) using real Ollama models. It ingests the documents into a temporary profile, asks a fixed question set and reports:

- pass rate;
- wrong refusals (in-scope questions the bot declined);
- whether answers cite the expected source;
- mean best retrieval score;
- whether the answer came back in the expected language.

Each question file ([English](../samples/eval_questions.json), [Malay](../samples/eval_questions_ms.json), [Chinese](../samples/eval_questions_zh.json)) has 15 questions: 10 answerable from the documents and 5 that should be refused.

```powershell
python samples/run_eval.py                                          # English
python samples/run_eval.py --questions samples/eval_questions_ms.json  # Malay
python samples/run_eval.py --all                                    # English, Malay and Chinese
python samples/run_eval.py --embedding-model bge-m3 --all           # Compare another embedding model
```

Other options: `--chat-model <tag>` and `--threshold <0–1>` override the [profile defaults](configuration.md#profile-settings), which makes it easy to compare models or tune the refusal threshold.

[`samples/generate_samples.py`](../samples/generate_samples.py) regenerates the sample documents.

---

[← README](../README.md) · [Docs](README.md) · [Architecture](architecture.md) · [Configuration](configuration.md) · [Languages](languages.md) · [API reference](api.md) · **Testing**
