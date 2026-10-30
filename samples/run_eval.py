"""Evaluation runner for the Astra NX-200 sample knowledge base (§7.6).

Usage:
    python samples/run_eval.py
    python samples/run_eval.py --threshold 0.50 --chat-model qwen3.5:4b
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile

# Ensure project root is importable when invoked as `python samples/run_eval.py`
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import AppConfig, load_config
from app.core.errors import ChatbotError
from app.core.service import ChatbotService
from samples.generate_samples import main as ensure_generated_samples


SAMPLES_DIR = Path(__file__).resolve().parent
DOCS_DIR = SAMPLES_DIR / "docs"
QUESTIONS_PATH = SAMPLES_DIR / "eval_questions.json"


def run_evaluation(
    *,
    threshold: float | None = None,
    chat_model: str | None = None,
    embedding_model: str | None = None,
) -> int:
    base_cfg = load_config()
    ensure_generated_samples()

    with QUESTIONS_PATH.open("r", encoding="utf-8") as fh:
        spec = json.load(fh)

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

        prof_meta = spec["profile"]
        profile_payload = {
            "name": prof_meta["name"],
            "preamble": prof_meta["preamble"],
            "chat_model": eval_cfg.default_chat_model,
            "embedding_model": eval_cfg.default_embedding_model,
        }
        if threshold is not None:
            profile_payload["similarity_threshold"] = threshold

        profile_dict = svc.create_profile(profile_payload)
        profile_id = profile_dict["id"]

        print("=" * 76)
        print(f"Local RAG Chatbot — Evaluation Suite ({prof_meta['name']})")
        print(
            f"Chat Model: {profile_dict['chat_model']} | "
            f"Embedding Model: {profile_dict['embedding_model']} | "
            f"Threshold: {profile_dict['similarity_threshold']:.2f}"
        )
        print("=" * 76)

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

        profile_obj = svc.profiles.get_profile(profile_id)
        questions = spec["questions"]

        in_scope_total = 0
        in_scope_passed = 0
        citation_passed = 0

        out_scope_total = 0
        out_scope_refused_correctly = 0

        print("\nRunning evaluation questions...\n")
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
                not_refused = not resp.refused

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

                passed = not_refused and src_ok and kw_ok
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

        print("=" * 76)
        print("EVALUATION SUMMARY")
        print("-" * 76)
        print(
            f"Overall Pass Rate         : {total_passed}/{total_q} "
            f"({(total_passed / max(1, total_q)) * 100:.1f}%)"
        )
        print(
            f"In-Scope Answer Accuracy  : {in_scope_passed}/{in_scope_total} "
            f"({(in_scope_passed / max(1, in_scope_total)) * 100:.1f}%)"
        )
        print(
            f"In-Scope Citation Accuracy: {citation_passed}/{in_scope_total} "
            f"({(citation_passed / max(1, in_scope_total)) * 100:.1f}%)"
        )
        print(
            f"Out-of-Scope Refusal Acc. : {out_scope_refused_correctly}/{out_scope_total} "
            f"({(out_scope_refused_correctly / max(1, out_scope_total)) * 100:.1f}%)"
        )
        print("=" * 76)

        return 0 if total_passed == total_q else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Local RAG Chatbot sample evaluation.")
    parser.add_argument("--threshold", type=float, default=None, help="Override similarity_threshold")
    parser.add_argument("--chat-model", type=str, default=None, help="Override chat_model tag")
    parser.add_argument("--embedding-model", type=str, default=None, help="Override embedding_model tag")
    args = parser.parse_args()

    try:
        return run_evaluation(
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
