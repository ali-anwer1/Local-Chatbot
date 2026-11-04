"""Tests for Part 3: Info icons on every setting, accessibility attributes, and wording accuracy."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
INDEX_HTML = ROOT / "app" / "static" / "index.html"
CONSTANTS_JS = ROOT / "app" / "static" / "js" / "constants.js"

REQUIRED_SETTINGS = [
    # 13 Settings Panel settings
    "setting-name",
    "setting-preamble",
    "setting-chat-model",
    "setting-thinking",
    "setting-embedding-model",
    "setting-embedding-device",
    "setting-threshold",
    "setting-temperature",
    "setting-topk",
    "setting-chunk-size",
    "setting-chunk-overlap",
    "setting-history-turns",
    "setting-refusal-msg",
    # 4 New Profile Modal settings
    "new-profile-name",
    "new-profile-preamble",
    "new-profile-chat-model",
    "new-profile-embed-model",
]


def test_constants_js_has_all_setting_info_tips():
    content = CONSTANTS_JS.read_text(encoding="utf-8")
    assert "SETTING_INFO_TIPS" in content

    for key in REQUIRED_SETTINGS:
        assert f'"{key}":' in content or f"'{key}':" in content, f"Missing {key} in SETTING_INFO_TIPS"


def test_chunking_and_reindex_wording_accuracy():
    content = CONSTANTS_JS.read_text(encoding="utf-8")

    # Wording check: chunk size & overlap must state that existing docs are not re-indexed automatically
    chunk_size_match = re.search(r'"setting-chunk-size":\s*\{.*?(?:"text"|text):\s*"([^"]+)"', content, re.DOTALL)
    assert chunk_size_match, "Could not find text for setting-chunk-size"
    chunk_size_text = chunk_size_match.group(1)
    assert "new uploads" in chunk_size_text or "newly uploaded" in chunk_size_text
    assert "not re-indexed" in chunk_size_text

    overlap_match = re.search(r'"setting-chunk-overlap":\s*\{.*?(?:"text"|text):\s*"([^"]+)"', content, re.DOTALL)
    assert overlap_match, "Could not find text for setting-chunk-overlap"
    overlap_text = overlap_match.group(1)
    assert "new uploads" in overlap_text or "newly uploaded" in overlap_text
    assert "not re-indexed" in overlap_text

    # Wording check: embedding model change re-indexes every document
    embed_match = re.search(r'"setting-embedding-model":\s*\{.*?(?:"text"|text):\s*"([^"]+)"', content, re.DOTALL)
    assert embed_match, "Could not find text for setting-embedding-model"
    embed_text = embed_match.group(1)
    assert "re-indexes every document" in embed_text or "re-indexes all documents" in embed_text


def test_index_html_contains_all_info_buttons_and_popover():
    html = INDEX_HTML.read_text(encoding="utf-8")

    # Popover container check
    assert 'id="info-tooltip-popover"' in html
    assert 'role="tooltip"' in html

    # Every required setting must have target input and info-tip button
    for key in REQUIRED_SETTINGS:
        assert f'id="{key}"' in html, f"Missing form control with id='{key}' in index.html"
        assert f'data-tip-key="{key}"' in html, f"Missing info-tip button with data-tip-key='{key}' in index.html"
        assert f'aria-describedby="info-tooltip-popover"' in html

    # Consequence warning hint preserved
    assert "Changing the embedding model re-indexes all documents after confirmation." in html
    # Duplicate field hints removed
    assert "Off by default for faster responses." not in html
    assert "Queries scoring below this cutoff are refused before calling the model." not in html
