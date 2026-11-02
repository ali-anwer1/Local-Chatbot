"""Evaluation runner for the Astra NX-200 sample knowledge base (§7.6 + Multilingual Part 1).

Usage:
    python samples/run_eval.py
    python samples/run_eval.py --questions samples/eval_questions_ms.json
    python samples/run_eval.py --embedding-model bge-m3
    python samples/run_eval.py --all
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile
from typing import Any

# Ensure project root is importable when invoked as `python samples/run_eval.py`
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import AppConfig, load_config
from app.core.errors import ChatbotError
from app.core.languages import detect_language
from app.core.service import ChatbotService
from samples.generate_samples import main as ensure_generated_samples


SAMPLES_DIR = Path(__file__).resolve().parent
DOCS_DIR = SAMPLES_DIR / "docs"
DEFAULT_QUESTIONS_PATH = SAMPLES_DIR / "eval_questions.json"
MS_QUESTIONS_PATH = SAMPLES_DIR / "eval_questions_ms.json"
ZH_QUESTIONS_PATH = SAMPLES_DIR / "eval_questions_zh.json"


def evaluate_single_file(
    svc: ChatbotService,
    profile_id: str,
    questions_path: Path,
) -> dict[str, Any]:
    """Run evaluation against a single questions file and collect metrics."""
    with questions_path.open("r", encoding="utf-8") as fh:
        spec = json.load(fh)

    expected_lang = "en"
    if "_ms.json" in questions_path.name:
        expected_lang = "ms"
    elif "_zh.json" in questions_path.name:
        expected_lang = "zh"

    profile_obj = svc.profiles.get_profile(profile_id)
    questions = spec["questions"]

    in_scope_total = 0
    in_scope_passed = 0
    citation_passed = 0
    wrong_refusals = 0
    in_scope_scores: list[float] = []
    lang_matches = 0
    in_scope_answered = 0

    out_scope_total = 0
    out_scope_refused_correctly = 0

    print(f"\n--- Running evaluation for {questions_path.name} (Expected Lang: {expected_lang}) ---")

    for item in questions:
        qid = item["id"]
        question = item["question"]
        expect_refusal = bool(item["expected_refusal"])

        resp = svc.rag.query(profile=profile_obj, question=question, history=[])

        if expect_refusal:
            out_scope_total += 1
            passed = resp.refused
            if passed:
                out_scope_refused_correctly += 1
            status_str = "PASS" if passed else "FAIL (should have refused)"
            print(
                f"[{qid}] {status_str} | best_score={resp.best_score:.3f} | refused={resp.refused}\n"
                f"  Q: {question}\n"
                f"  A: {resp.answer[:160]}\n"
            )
        else:
            in_scope_total += 1
            in_scope_scores.append(resp.best_score)
            not_refused = not resp.refused

            if not not_refused:
                wrong_refusals += 1

            expected_src = item.get("expected_source_file")
            expected_page = item.get("expected_page")
            retrieved_files = {s["source_file"] for s in resp.sources}
            src_ok = (expected_src in retrieved_files) if expected_src else True
            if expected_page is not None:
                src_ok = any(
                    s["source_file"] == expected_src and s.get("page") == expected_page
                    for s in resp.sources
                )
            has_citation_marker = "[" in resp.answer and "]" in resp.answer
            cite_ok = src_ok and has_citation_marker
            if cite_ok and not_refused:
                citation_passed += 1

            keywords = item.get("expected_keywords", [])
            answer_lower = resp.answer.lower()
            matched_kw = sum(1 for kw in keywords if kw.lower() in answer_lower)
            kw_ok = matched_kw >= max(1, len(keywords) - 1) if keywords else True

            # Language check on non-refused answers
            lang_ok = True
            if not_refused and resp.answer:
                in_scope_answered += 1
                det_code, _ = detect_language(resp.answer)
                if det_code == expected_lang:
                    lang_matches += 1
                else:
                    lang_ok = False

            passed = not_refused and src_ok and kw_ok and lang_ok
            if passed:
                in_scope_passed += 1

            status_str = "PASS" if passed else "FAIL"
            sources_summary = ", ".join(
                f"[{s['index']}] {s['source_file']}"
                + (f" ({s['location_label']})" if s.get("location_label") else "")
                for s in resp.sources[:3]
            )
            print(
                f"[{qid}] {status_str} | best_score={resp.best_score:.3f} | keywords={matched_kw}/{len(keywords)}\n"
                f"  Q: {question}\n"
                f"  Sources: {sources_summary or 'None'}\n"
                f"  A: {resp.answer[:180].replace(chr(10), ' ')}\n"
            )

    total_q = in_scope_total + out_scope_total
    total_passed = in_scope_passed + out_scope_refused_correctly
    mean_score = sum(in_scope_scores) / max(1, len(in_scope_scores))
    lang_rate_str = (
        f"{lang_matches}/{in_scope_answered} ({(lang_matches / max(1, in_scope_answered)) * 100:.0f}%)"
        if in_scope_answered > 0
        else "N/A"
    )

    return {
        "file": questions_path.name,
        "total_passed": total_passed,
        "total_q": total_q,
        "pass_rate_pct": (total_passed / max(1, total_q)) * 100,
        "mean_best_score": mean_score,
        "wrong_refusals": wrong_refusals,
        "in_scope_total": in_scope_total,
        "lang_matches": lang_matches,
        "in_scope_answered": in_scope_answered,
        "lang_rate_str": lang_rate_str,
    }


def run_evaluation(
    *,
    questions_file: str | None = None,
    run_all: bool = False,
    threshold: float | None = None,
    chat_model: str | None = None,
    embedding_model: str | None = None,
) -> int:
    base_cfg = load_config()
    ensure_generated_samples()

    if run_all:
        files_to_run = [DEFAULT_QUESTIONS_PATH, MS_QUESTIONS_PATH, ZH_QUESTIONS_PATH]
    elif questions_file:
        files_to_run = [Path(questions_file)]
    else:
        files_to_run = [DEFAULT_QUESTIONS_PATH]

    with tempfile.TemporaryDirectory(prefix="local_rag_eval_") as tmp_dir:
        eval_cfg = AppConfig(
            host="127.0.0.1",
            port=base_cfg.port,
            data_dir=Path(tmp_dir),
            ollama_base_url=base_cfg.ollama_base_url,
            default_chat_model=chat_model or base_cfg.default_chat_model,
            default_embedding_model=embedding_model or base_cfg.default_embedding_model,
            allow_chat_fallback=base_cfg.allow_chat_fallback,
        )
        svc = ChatbotService(config=eval_cfg)

        health = svc.get_health(force_refresh=True)
        if not health["ollama_reachable"]:
            print(f"[ERROR] {health['banner_message']}")
            print("Start Ollama locally and ensure the configured models are pulled before running live eval.")
            return 2
        if health["no_models_installed"]:
            print("[ERROR] No Ollama models installed. Run:")
            for cmd in health["recommended_commands"]:
                print(f"  {cmd}")
            return 2

        profile_payload = {
            "name": "Astra NX-200 Technical Assistant",
            "preamble": "You are an internal hardware engineering and field-support assistant for the Astra NX-200 Industrial Edge Gateway. Provide accurate, concise technical answers strictly from the provided documentation and always cite your source passages.",
            "chat_model": eval_cfg.default_chat_model,
            "embedding_model": eval_cfg.default_embedding_model,
        }
        if threshold is not None:
            profile_payload["similarity_threshold"] = threshold

        profile_dict = svc.create_profile(profile_payload)
        profile_id = profile_dict["id"]

        print("=" * 84)
        print("Local RAG Chatbot — Multilingual Evaluation Suite")
        print(
            f"Chat Model: {profile_dict['chat_model']} | "
            f"Embedding Model: {profile_dict['embedding_model']} | "
            f"Threshold: {profile_dict['similarity_threshold']:.2f}"
        )
        print("=" * 84)

        doc_files = sorted(
            p for p in DOCS_DIR.iterdir() if p.is_file() and p.suffix.lower() in {".pdf", ".docx", ".md", ".txt"}
        )
        print(f"\nIngesting {len(doc_files)} sample documents...")
        for doc_path in doc_files:
            res = svc.ingest_document(
                profile_id,
                filename=doc_path.name,
                data=doc_path.read_bytes(),
            )
            doc_info = res["document"]
            print(f"  - {doc_info['filename']}: {doc_info['chunk_count']} chunk(s) [{doc_info['status']}]")

        summaries: list[dict[str, Any]] = []
        all_passed = True
        for q_file in files_to_run:
            if not q_file.exists():
                print(f"[WARN] File not found: {q_file}")
                continue
            summary = evaluate_single_file(svc, profile_id, q_file)
            summaries.append(summary)
            if summary["total_passed"] < summary["total_q"]:
                all_passed = False

        # Summary Table
        print("\n" + "=" * 84)
        print("MULTILINGUAL RETRIEVAL & ACCURACY SUMMARY TABLE")
        print("-" * 84)
        print(f"{'Questions File':<26} {'Pass Rate':<18} {'Mean In-Scope Score':<22} {'Wrong Refusals':<16} {'Expected Lang'}")
        print("-" * 84)
        for s in summaries:
            pass_str = f"{s['total_passed']}/{s['total_q']} ({s['pass_rate_pct']:.1f}%)"
            score_str = f"{s['mean_best_score']:.3f}"
            refusal_str = f"{s['wrong_refusals']}/{s['in_scope_total']}"
            print(f"{s['file']:<26} {pass_str:<18} {score_str:<22} {refusal_str:<16} {s['lang_rate_str']}")
        print("=" * 84)

        # One-line comparison note
        print(
            "\n[Comparison] English-centric models like 'nomic-embed-text' score cross-lingual queries "
            "lower against English passages. If non-English questions yield unwanted refusals, switch to a "
            "multilingual embedding model (e.g. 'bge-m3') in Settings without changing application code."
        )

        return 0 if all_passed else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Local RAG Chatbot sample evaluation.")
    parser.add_argument("--questions", "--questions-file", "-q", dest="questions_file", type=str, default=None, help="Path to questions JSON file")
    parser.add_argument("--all", action="store_true", help="Evaluate en, ms, and zh question files")
    parser.add_argument("--threshold", type=float, default=None, help="Override similarity_threshold")
    parser.add_argument("--chat-model", type=str, default=None, help="Override chat_model tag")
    parser.add_argument("--embedding-model", type=str, default=None, help="Override embedding_model tag")
    args = parser.parse_args()

    try:
        return run_evaluation(
            questions_file=args.questions_file,
            run_all=args.all,
            threshold=args.threshold,
            chat_model=args.chat_model,
            embedding_model=args.embedding_model,
        )
    except ChatbotError as exc:
        print(f"[ERROR] ({exc.code}) {exc.message}")
        if exc.details:
            print(f"Details: {exc.details}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
