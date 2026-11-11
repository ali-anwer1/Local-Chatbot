[← README](../README.md) · [Docs](README.md) · [Architecture](architecture.md) · [Configuration](configuration.md) · [Languages](languages.md) · **API reference** · [Testing](testing.md)

# API reference

All endpoints are served from `http://127.0.0.1:8000`. FastAPI's interactive docs are at `/api/docs` (they load their page assets from a CDN, so they need an internet connection).

Streaming endpoints return [Server-Sent Events](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events). Errors are returned as JSON with `code`, `message` and optional `details`.

Routes are defined in [`app/main.py`](../app/main.py) and tested in [`test_availability_and_api.py`](../tests/test_availability_and_api.py).

## App

| Method | Path | Description |
| :--- | :--- | :--- |
| `GET` | `/api/health` | Ollama status and installed models. `?refresh=true` bypasses the cache |
| `GET` | `/api/app-settings` | Machine settings (`embedding_device`, `keep_alive`) |
| `PUT` | `/api/app-settings` | Set `embedding_device` to `cpu` or `auto` |
| `GET` | `/api/languages` | Supported languages |
| `POST` | `/api/models/pull` | Download a model through Ollama (SSE). Body: `{"model": "...", "confirmed": true}` |

## Profiles

| Method | Path | Description |
| :--- | :--- | :--- |
| `GET` | `/api/profiles` | List profiles with their readiness status |
| `POST` | `/api/profiles` | Create a profile |
| `GET` | `/api/profiles/{id}` | Profile settings, status and change history |
| `PUT` | `/api/profiles/{id}` | Update settings. Changing the embedding model needs `"confirm_reindex": true` |
| `DELETE` | `/api/profiles/{id}` | Delete the profile, its documents and its vectors |
| `POST` | `/api/profiles/{id}/reindex` | Re-embed all documents |

Profile fields are listed in [Configuration](configuration.md#profile-settings).

## Documents

| Method | Path | Description |
| :--- | :--- | :--- |
| `GET` | `/api/profiles/{id}/documents` | List documents |
| `POST` | `/api/profiles/{id}/documents` | Upload one file (`multipart/form-data`, field `file`): PDF, DOCX, MD or TXT |
| `DELETE` | `/api/profiles/{id}/documents/{doc_id}` | Remove a document and its vectors |

## Chat

| Method | Path | Description |
| :--- | :--- | :--- |
| `POST` | `/api/profiles/{id}/chat` | Ask a question (SSE) |
| `POST` | `/api/translate` | Translate an answer (SSE) |
| `GET` | `/api/profiles/{id}/chats` | List saved chats |
| `POST` | `/api/profiles/{id}/chats` | Save a chat (pass `id` to overwrite) |
| `GET` | `/api/profiles/{id}/chats/{chat_id}` | Load a saved chat |
| `DELETE` | `/api/profiles/{id}/chats/{chat_id}` | Delete a saved chat |

### `POST /api/profiles/{id}/chat`

```json
{
  "message": "What does fault code E-409 mean?",
  "history": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}],
  "options": {"think": false},
  "requested_language": "ms"
}
```

Only `message` is required. The server keeps no conversation state; the client sends `history` with each request. `requested_language` is the composer's "Answer in" choice ([how it's used](languages.md#how-the-answer-language-is-chosen)). For what happens between request and answer, see [Architecture](architecture.md#how-a-question-is-answered).

| Event | Contents |
| :--- | :--- |
| `status` | Progress label, e.g. retrieving or generating |
| `meta` | Model used, fallback status, best retrieval score, language decision |
| `token` | `{"text": "..."}` – the next piece of the answer |
| `done` | Full text, `refused`, cited `sources`, `timings`, `language` |
| `error` | `code`, `message`, `details` |

### `POST /api/translate`

```json
{"text": "...", "target_language": "zh", "profile_id": "astra-nx-200-spec-assistant", "source_language": "en"}
```

Streams `status`, `token`, `done` and `error` events like the chat endpoint.

---

[← README](../README.md) · [Docs](README.md) · [Architecture](architecture.md) · [Configuration](configuration.md) · [Languages](languages.md) · **API reference** · [Testing](testing.md)
