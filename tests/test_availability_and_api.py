"""Tests covering every row of §4.7 model availability/fallbacks, OllamaBackend HTTP logic, and FastAPI routes (§7.8, §7.9)."""

from __future__ import annotations

import json
from pathlib import Path
from fastapi.testclient import TestClient
import httpx

from app.config import AppConfig
from app.core.backends.ollama import OllamaBackend
from app.core.service import ChatbotService
from app.main import create_app
from tests.conftest import DeterministicMockBackend


def test_row1_ollama_unreachable_handled_gracefully(
    test_config: AppConfig,
    mock_backend: DeterministicMockBackend,
) -> None:
    """§4.7 Row 1: Ollama not running -> global banner, chat & upload blocked, no crash."""
    mock_backend.reachable = False
    svc = ChatbotService(config=test_config, backend=mock_backend)
    client = TestClient(create_app(svc))

    health = client.get("/api/health").json()
    assert health["ollama_reachable"] is False
    assert "Ollama is not reachable at http://127.0.0.1:11434" in health["banner_message"]

    prof = client.post("/api/profiles", json={"name": "Offline Test"}).json()
    pid = prof["id"]
    assert prof["availability"]["chat_blocked"] is True
    assert prof["availability"]["upload_blocked"] is True

    chat_res = client.post(f"/api/profiles/{pid}/chat", json={"message": "Hello"})
    assert chat_res.status_code == 503
    assert chat_res.json()["code"] == "ollama_unreachable"
    assert "Traceback" not in chat_res.text

    up_res = client.post(
        f"/api/profiles/{pid}/documents",
        files={"file": ("note.txt", b"hello world", "text/plain")},
    )
    assert up_res.status_code == 503
    assert up_res.json()["code"] == "ollama_unreachable"


def test_row2_no_models_installed_returns_first_run_commands(
    test_config: AppConfig,
) -> None:
    """§4.7 Row 2: 0 models installed -> first-run status with exact ollama pull commands."""
    empty_backend = DeterministicMockBackend(models=[])
    svc = ChatbotService(config=test_config, backend=empty_backend)
    client = TestClient(create_app(svc))

    health = client.get("/api/health").json()
    assert health["ollama_reachable"] is True
    assert health["no_models_installed"] is True
    assert "ollama pull qwen3.5:4b" in health["recommended_commands"]
    assert "ollama pull nomic-embed-text" in health["recommended_commands"]


def test_row3a_chat_model_missing_with_fallback_enabled(
    test_config: AppConfig,
    mock_backend: DeterministicMockBackend,
) -> None:
    """§4.7 Row 3a: Chat model missing + allow_chat_fallback=True -> uses default with explicit banner."""
    svc = ChatbotService(config=test_config, backend=mock_backend)
    client = TestClient(create_app(svc))

    prof = client.post(
        "/api/profiles",
        json={"name": "Missing Chat Bot", "chat_model": "deleted-model:7b"},
    ).json()
    pid = prof["id"]

    avail = prof["availability"]
    assert avail["has_warning"] is True
    assert avail["chat_blocked"] is False
    assert avail["fallback_active"] is True
    assert avail["fallback_banner"] == "Using qwen3.5:4b because deleted-model:7b is not installed."

    client.post(
        f"/api/profiles/{pid}/documents",
        files={"file": ("spec.txt", b"Warranty is 36-month standard coverage.", "text/plain")},
    )
    chat_resp = client.post(
        f"/api/profiles/{pid}/chat",
        json={"message": "What is the 36-month warranty?"},
    )
    assert chat_resp.status_code == 200
    assert '"fallback_active": true' in chat_resp.text
    assert "Using qwen3.5:4b because deleted-model:7b is not installed." in chat_resp.text


def test_row3b_chat_model_missing_with_fallback_disabled_blocks_chat(
    tmp_path: Path,
    mock_backend: DeterministicMockBackend,
) -> None:
    """§4.7 Row 3b: Chat model missing + allow_chat_fallback=False -> blocks chat, never substitutes silently."""
    cfg = AppConfig(
        data_dir=tmp_path / "data_nofallback",
        default_chat_model="qwen3.5:4b",
        default_embedding_model="nomic-embed-text",
        allow_chat_fallback=False,
    )
    svc = ChatbotService(config=cfg, backend=mock_backend)
    client = TestClient(create_app(svc))

    prof = client.post(
        "/api/profiles",
        json={"name": "Blocked Chat Bot", "chat_model": "missing-qwen:14b"},
    ).json()
    pid = prof["id"]
    assert prof["availability"]["chat_blocked"] is True
    assert prof["availability"]["block_code"] == "chat_model_missing"

    res = client.post(f"/api/profiles/{pid}/chat", json={"message": "Hi"})
    assert res.status_code == 409
    body = res.json()
    assert body["code"] == "chat_model_missing"
    assert "qwen3.5:4b" in body["available_chat_models"]
    assert len(mock_backend.chat_calls) == 0


def test_row4_embedding_model_missing_never_falls_back(
    test_config: AppConfig,
    mock_backend: DeterministicMockBackend,
) -> None:
    """§4.7 Row 4: Embedding model missing -> NO fallback; both chat and upload blocked."""
    svc = ChatbotService(config=test_config, backend=mock_backend)
    client = TestClient(create_app(svc))

    prof = client.post(
        "/api/profiles",
        json={"name": "Missing Embed Bot", "embedding_model": "snowflake-arctic-embed:latest"},
    ).json()
    pid = prof["id"]
    assert prof["availability"]["chat_blocked"] is True
    assert prof["availability"]["upload_blocked"] is True
    assert prof["availability"]["block_code"] == "embedding_model_missing"

    up_res = client.post(
        f"/api/profiles/{pid}/documents",
        files={"file": ("doc.txt", b"Warranty is 36 months.", "text/plain")},
    )
    assert up_res.status_code == 409
    assert up_res.json()["code"] == "embedding_model_missing"

    chat_res = client.post(f"/api/profiles/{pid}/chat", json={"message": "Warranty?"})
    assert chat_res.status_code == 409
    assert chat_res.json()["code"] == "embedding_model_missing"


def test_row5_collection_metadata_mismatch_blocks_until_reindexed(
    test_config: AppConfig,
    mock_backend: DeterministicMockBackend,
) -> None:
    """§4.7 Row 5: Hand-edited profile.json or dimension mismatch blocks chat/upload and offers re-index."""
    svc = ChatbotService(config=test_config, backend=mock_backend)
    client = TestClient(create_app(svc))

    prof = client.post("/api/profiles", json={"name": "Tamper Test"}).json()
    pid = prof["id"]
    client.post(
        f"/api/profiles/{pid}/documents",
        files={"file": ("doc.txt", b"Input voltage is 12 V to 48 V DC power.", "text/plain")},
    )

    # Simulate hand-editing profile.json on disk to a different embedding model without re-indexing
    profile_json_path = test_config.data_dir / "profiles" / pid / "profile.json"
    raw = json.loads(profile_json_path.read_text(encoding="utf-8"))
    raw["embedding_model"] = "mxbai-embed-large"
    profile_json_path.write_text(json.dumps(raw), encoding="utf-8")

    inspected = client.get(f"/api/profiles/{pid}").json()
    assert inspected["availability"]["metadata_ok"] is False
    assert inspected["availability"]["chat_blocked"] is True
    assert inspected["availability"]["upload_blocked"] is True
    assert inspected["availability"]["block_code"] == "collection_metadata_mismatch"

    blocked_chat = client.post(
        f"/api/profiles/{pid}/chat", json={"message": "What is the DC voltage?"}
    )
    assert blocked_chat.status_code == 409
    assert blocked_chat.json()["code"] == "collection_metadata_mismatch"

    # Explicit re-index repairs collection metadata and restores retrieval
    reindexed = client.post(f"/api/profiles/{pid}/reindex")
    assert reindexed.status_code == 200
    assert reindexed.json()["availability"]["metadata_ok"] is True

    ok_chat = client.post(
        f"/api/profiles/{pid}/chat", json={"message": "What is the 12 V to 48 V DC voltage?"}
    )
    assert ok_chat.status_code == 200
    assert '"refused": false' in ok_chat.text


def test_row6_model_load_oom_failure_returns_friendly_message_and_details(
    test_config: AppConfig,
    mock_backend: DeterministicMockBackend,
) -> None:
    """§4.7 Row 6: OOM / load failure caught cleanly both pre-stream and mid-stream with expandable details."""
    svc = ChatbotService(config=test_config, backend=mock_backend)
    client = TestClient(create_app(svc))

    prof = client.post("/api/profiles", json={"name": "OOM Bot"}).json()
    pid = prof["id"]
    client.post(
        f"/api/profiles/{pid}/documents",
        files={"file": ("doc.txt", b"Warranty is 36-month hardware coverage.", "text/plain")},
    )

    # Case A: Embedding model fails to load (pre-stream)
    mock_backend.fail_load_models.add("nomic-embed-text:latest")
    res_pre = client.post(
        f"/api/profiles/{pid}/chat", json={"message": "What is the 36-month warranty?"}
    )
    assert res_pre.status_code == 500
    body_pre = res_pre.json()
    assert body_pre["code"] == "model_load_failed"
    assert "failed to load, likely insufficient memory. Try a smaller model." in body_pre["message"]
    assert "failed to allocate" in body_pre["details"]
    assert "Traceback" not in res_pre.text
    mock_backend.fail_load_models.clear()

    # Case B: Chat model fails mid-stream due to KV-cache OOM
    mock_backend.fail_mid_stream_models.add("qwen3.5:4b")
    res_mid = client.post(
        f"/api/profiles/{pid}/chat", json={"message": "What is the 36-month warranty?"}
    )
    assert res_mid.status_code == 200
    assert "event: error" in res_mid.text
    assert "qwen3.5:4b failed to load, likely insufficient memory. Try a smaller model." in res_mid.text
    assert "KV cache allocation" in res_mid.text
    assert "Traceback" not in res_mid.text


def test_offline_model_download_and_unconfirmed_guard(
    test_config: AppConfig,
    mock_backend: DeterministicMockBackend,
) -> None:
    """§4.7 Downloads: requires confirmation; fails gracefully offline."""
    svc = ChatbotService(config=test_config, backend=mock_backend)
    client = TestClient(create_app(svc))

    # Unconfirmed pull rejected
    unconfirmed = client.post(
        "/api/models/pull", json={"model": "qwen3.5:4b", "confirmed": False}
    )
    assert unconfirmed.status_code == 422

    # Offline pull fails with exact required message
    mock_backend.pull_offline = True
    offline_res = client.post(
        "/api/models/pull", json={"model": "qwen3.5:4b", "confirmed": True}
    )
    assert offline_res.status_code == 503
    assert offline_res.json()["message"] == "No internet connection; download unavailable offline"


def test_ollama_backend_capability_detection_and_heuristic_fallback() -> None:
    """§4.4: OllamaBackend reads /api/show capabilities and falls back to name heuristic ('unverified')."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/tags":
            return httpx.Response(
                200,
                json={
                    "models": [
                        {"name": "qwen3.5:4b", "size": 2700000000, "digest": "d1"},
                        {"name": "legacy-bge-small:latest", "size": 130000000, "digest": "d2"},
                        {"name": "legacy-mistral:latest", "size": 4100000000, "digest": "d3"},
                    ]
                },
            )
        if request.url.path == "/api/show":
            body = json.loads(request.content.decode("utf-8"))
            if body["model"] == "qwen3.5:4b":
                return httpx.Response(
                    200,
                    json={"capabilities": ["completion", "tools", "thinking"]},
                )
            # Simulate older Ollama omitting capabilities field
            return httpx.Response(200, json={"details": {"parameter_size": "7B"}})
        return httpx.Response(404)

    backend = OllamaBackend(transport=httpx.MockTransport(handler))
    models = backend.list_models()
    by_name = {m["name"]: m for m in models}

    assert by_name["qwen3.5:4b"]["capabilities"] == ["completion"]
    assert by_name["qwen3.5:4b"]["capabilities_verified"] is True

    assert by_name["legacy-bge-small:latest"]["capabilities"] == ["embedding"]
    assert by_name["legacy-bge-small:latest"]["capabilities_verified"] is False

    assert by_name["legacy-mistral:latest"]["capabilities"] == ["completion"]
    assert by_name["legacy-mistral:latest"]["capabilities_verified"] is False


def test_ollama_backend_disables_thinking_by_default_and_retries_legacy_400() -> None:
    """OllamaBackend sends think=False by default and retries without 'think' if legacy server rejects it."""
    recorded_payloads: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/chat":
            payload = json.loads(request.content.decode("utf-8"))
            recorded_payloads.append(payload)
            if payload.get("model") == "legacy-model" and "think" in payload:
                return httpx.Response(400, json={"error": "unknown field think"})
            return httpx.Response(
                200,
                json={"message": {"role": "assistant", "content": "Fast reply"}, "done": True},
            )
        return httpx.Response(404)

    backend = OllamaBackend(transport=httpx.MockTransport(handler))

    # 1. Default options -> think=False sent explicitly even before list_models() cache exists
    out1 = "".join(
        backend.chat(
            messages=[{"role": "user", "content": "Hello"}],
            model="qwen3.5:4b",
            options={"temperature": 0.2},
            stream=False,
        )
    )
    assert out1 == "Fast reply"
    assert recorded_payloads[0]["think"] is False

    # 2. Explicit think=True
    "".join(
        backend.chat(
            messages=[{"role": "user", "content": "Hello"}],
            model="qwen3.5:4b",
            options={"think": True},
            stream=False,
        )
    )
    assert recorded_payloads[1]["think"] is True

    # 3. Legacy model returning 400 on 'think' automatically retries without 'think'
    out3 = "".join(
        backend.chat(
            messages=[{"role": "user", "content": "Hello"}],
            model="legacy-model",
            options={"think": False},
            stream=False,
        )
    )
    assert out3 == "Fast reply"
    assert "think" in recorded_payloads[2]
    assert "think" not in recorded_payloads[3]


def test_saved_chats_and_retry_history_api_routes(
    test_config: AppConfig,
    mock_backend: DeterministicMockBackend,
) -> None:
    """HTTP endpoints for saving, listing, loading, and deleting saved chats work end-to-end."""
    svc = ChatbotService(config=test_config, backend=mock_backend)
    client = TestClient(create_app(svc))

    prof = client.post("/api/profiles", json={"name": "API Saved Chats"}).json()
    pid = prof["id"]

    save_res = client.post(
        f"/api/profiles/{pid}/chats",
        json={
            "title": "Maintenance Q&A",
            "turns": [
                {"role": "user", "text": "What battery is used?"},
                {"role": "assistant", "text": "CR2032W coin cell [1]."},
            ],
        },
    )
    assert save_res.status_code == 201
    chat_id = save_res.json()["id"]

    list_res = client.get(f"/api/profiles/{pid}/chats")
    assert list_res.status_code == 200
    assert len(list_res.json()["chats"]) == 1

    # Legacy ?session_id=... query parameter remains accepted (no 422) and ignored
    get_legacy = client.get(f"/api/profiles/{pid}/chats/{chat_id}?session_id=api-sess")
    assert get_legacy.status_code == 200
    assert get_legacy.json()["title"] == "Maintenance Q&A"

    get_res = client.get(f"/api/profiles/{pid}/chats/{chat_id}")
    assert get_res.status_code == 200
    assert get_res.json()["title"] == "Maintenance Q&A"

    # Legacy DELETE /history remains a no-op returning the same JSON shape
    clear_res = client.delete(f"/api/profiles/{pid}/history?session_id=api-sess")
    assert clear_res.status_code == 200
    assert clear_res.json() == {
        "cleared": True,
        "profile_id": pid,
        "session_id": "api-sess",
    }

    # Client-supplied history in POST /chat reaches the assembled LLM prompt
    client.post(
        f"/api/profiles/{pid}/documents",
        files={
            "file": (
                "rtc.txt",
                b"The RTC battery is a CR2032W coin cell on a preventive schedule.",
                "text/plain",
            )
        },
    )
    chat_res = client.post(
        f"/api/profiles/{pid}/chat",
        json={
            "message": "What preventive schedule applies to the RTC battery?",
            "session_id": "unused-legacy-id",
            "history": [
                {"role": "user", "content": "What battery is used?"},
                {"role": "assistant", "content": "CR2032W coin cell [1]."},
            ],
        },
    )
    assert chat_res.status_code == 200
    sent_messages = mock_backend.chat_calls[-1]["messages"]
    assert {"role": "user", "content": "What battery is used?"} in sent_messages
    assert {"role": "assistant", "content": "CR2032W coin cell [1]."} in sent_messages

    del_res = client.delete(f"/api/profiles/{pid}/chats/{chat_id}")
    assert del_res.status_code == 200
    assert client.get(f"/api/profiles/{pid}/chats").json()["chats"] == []
