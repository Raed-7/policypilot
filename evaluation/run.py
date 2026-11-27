"""Evaluate retrieval and answer quality on the labelled policy question set.

    python -m evaluation.run                # full grid + end-to-end evaluation
    python -m evaluation.run --quick        # default embedding model only
"""

import argparse
import itertools
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from evaluation.metrics import (
    fact_recall,
    first_relevant_rank,
    groundedness,
    mean_reciprocal_rank,
    percentile,
    recall_at_k,
)
from evaluation.report import write_report
from policypilot.chunking import chunk_document
from policypilot.config import ROOT_DIR, Settings
from policypilot.embeddings import SentenceTransformerEmbedder
from policypilot.generation import ExtractiveGenerator
from policypilot.ingestion import load_directory
from policypilot.llm import build_provider
from policypilot.pipeline import PolicyPilot
from policypilot.retrieval import Retriever
from policypilot.vectorstore import VectorStore

DATASET = Path(__file__).parent / "dataset.jsonl"
RESULTS = Path(__file__).parent / "results"
EMBEDDING_MODELS = ["BAAI/bge-small-en-v1.5", "sentence-transformers/all-MiniLM-L6-v2"]
STRATEGIES = ["section", "fixed"]
CHUNK_SIZES = [64, 128, 256]
OVERLAP_RATIOS = [0.0, 0.25]
MODES = ["dense", "hybrid"]
RETRIEVE_DEPTH = 10


def load_dataset(path: Path = DATASET) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def build_index(documents, embedder, strategy: str, size: int, overlap: int) -> tuple[Retriever, int, float]:
    started = time.perf_counter()
    store = VectorStore(None, collection_name=f"eval-{strategy}-{size}-{overlap}-{time.time_ns()}")
    chunks = [c for doc in documents for c in chunk_document(doc, size, overlap, strategy)]
    store.add(chunks, embedder.embed_documents([c.text for c in chunks]))
    return Retriever(store, embedder), len(chunks), time.perf_counter() - started


def evaluate_retrieval(retriever: Retriever, examples: list[dict], mode: str) -> dict:
    ranks, latencies = [], []
    for ex in examples:
        started = time.perf_counter()
        hits = retriever.retrieve(ex["question"], k=RETRIEVE_DEPTH, threshold=-1.0, mode=mode)
        latencies.append((time.perf_counter() - started) * 1000)
        ranks.append(first_relevant_rank([h.chunk for h in hits], ex))
    return {
        "recall@1": recall_at_k(ranks, 1),
        "recall@3": recall_at_k(ranks, 3),
        "recall@5": recall_at_k(ranks, 5),
        "mrr@10": mean_reciprocal_rank(ranks),
        "latency_ms_p50": percentile(latencies, 50),
        "misses@5": [ex["id"] for ex, r in zip(examples, ranks) if not r or r > 5],
    }


def _sweep(scores: np.ndarray, labels: np.ndarray) -> list[dict]:
    rows = []
    for t in np.round(np.arange(0.10, 0.90, 0.01), 2):
        accept = scores >= t
        tpr = float(np.mean(accept[labels]))
        tnr = float(np.mean(~accept[~labels]))
        rows.append({"threshold": float(t), "answer_rate": tpr, "refusal_rate": tnr, "balanced_accuracy": (tpr + tnr) / 2})
    return rows


def _auc(pos: np.ndarray, neg: np.ndarray) -> float:
    return float(np.mean([(p > n) + 0.5 * (p == n) for p in pos for n in neg]))


def calibrate_threshold(retriever: Retriever, examples: list[dict], mode: str, analyser: ExtractiveGenerator, top_k: int) -> dict:
    """Choose the answerability threshold that best separates answerable from unanswerable questions.

    Two signals are compared: the top chunk's cosine similarity, and the best sentence-level
    similarity inside the retrieved chunks (the signal the pipeline uses).
    """
    chunk_scores, sentence_scores, labels = [], [], []
    for ex in examples:
        hits = retriever.retrieve(ex["question"], k=top_k, threshold=-1.0, mode=mode)
        chunk_scores.append(max((h.dense_score for h in hits), default=0.0))
        sentence_scores.append(analyser.analyse(ex["question"], hits).best_similarity)
        labels.append(ex["answerable"])
    labels = np.array(labels)
    out = {}
    for name, scores in (("chunk", np.array(chunk_scores)), ("sentence", np.array(sentence_scores))):
        sweep = _sweep(scores, labels)
        best = max(sweep, key=lambda r: (round(r["balanced_accuracy"], 4), r["answer_rate"]))
        out[name] = {
            "best": best,
            "auc": _auc(scores[labels], scores[~labels]),
            "answerable": {"min": float(scores[labels].min()), "mean": float(scores[labels].mean())},
            "unanswerable": {"max": float(scores[~labels].max()), "mean": float(scores[~labels].mean())},
            "sweep": sweep,
        }
    return out


def evaluate_answers(pipeline: PolicyPilot, examples: list[dict], judge_embedder) -> dict:
    rows = []
    for ex in examples:
        answer = pipeline.ask(ex["question"])
        row = {
            "id": ex["id"],
            "question": ex["question"],
            "answerable": ex["answerable"],
            "predicted_answerable": answer.answerable,
            "answer": answer.answer,
            "provider": answer.provider,
            "latency_ms": answer.latency_ms["total"],
            "citations": [f"{c.source}{' p.' + str(c.page) if c.page else ''}" for c in answer.citations],
        }
        if ex["answerable"] and answer.answerable:
            g = groundedness(answer.answer, [c.chunk.text for c in answer.contexts], judge_embedder)
            row["fact_recall"] = fact_recall(answer.answer, ex["expected_facts"])
            row["sentence_support"] = g.sentence_support
            row["token_precision"] = g.token_precision
            row["citation_correct"] = any(
                c.source == ex["source"] and (ex.get("page") is None or c.page == ex["page"]) for c in answer.citations
            )
        rows.append(row)

    answerable = [r for r in rows if r["answerable"]]
    unanswerable = [r for r in rows if not r["answerable"]]
    answered = [r for r in answerable if r["predicted_answerable"]]
    mean = lambda key, items: float(np.mean([r[key] for r in items])) if items else 0.0
    return {
        "provider": pipeline.provider_name,
        "model": pipeline.model_name,
        "summary": {
            "answer_rate": len(answered) / len(answerable) if answerable else 0.0,
            "correct_refusal_rate": float(np.mean([not r["predicted_answerable"] for r in unanswerable])) if unanswerable else 0.0,
            "fact_recall": mean("fact_recall", answered),
            "fact_recall_all": float(np.mean([r.get("fact_recall", 0.0) for r in answerable])) if answerable else 0.0,
            "citation_accuracy": mean("citation_correct", answered),
            "sentence_support": mean("sentence_support", answered),
            "token_precision": mean("token_precision", answered),
            "latency_ms_p50": percentile([r["latency_ms"] for r in rows], 50),
            "latency_ms_p95": percentile([r["latency_ms"] for r in rows], 95),
        },
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--quick", action="store_true", help="evaluate only the default embedding model")
    parser.add_argument("--docs", type=Path, default=ROOT_DIR / "data" / "sample_policies")
    parser.add_argument("--out", type=Path, default=RESULTS)
    args = parser.parse_args()

    examples = load_dataset()
    answerable = [e for e in examples if e["answerable"]]
    documents = load_directory(args.docs)
    settings = Settings()
    models = EMBEDDING_MODELS[:1] if args.quick else EMBEDDING_MODELS
    print(f"{len(examples)} questions ({len(answerable)} answerable), {len(documents)} documents")

    grid = []
    embedders = {}
    for model in models:
        embedders[model] = embedder = SentenceTransformerEmbedder(model)
        for strategy, size, ratio in itertools.product(STRATEGIES, CHUNK_SIZES, OVERLAP_RATIOS):
            overlap = int(size * ratio)
            retriever, n_chunks, index_s = build_index(documents, embedder, strategy, size, overlap)
            for mode in MODES:
                metrics = evaluate_retrieval(retriever, answerable, mode)
                grid.append({
                    "embedding_model": model, "strategy": strategy, "chunk_size": size, "overlap": overlap,
                    "mode": mode, "chunks": n_chunks, "index_seconds": round(index_s, 2), **metrics,
                })
                print(f"{model.split('/')[-1]:>18} {strategy:>7} size={size:<4} overlap={overlap:<3} {mode:>6}  "
                      f"R@1={metrics['recall@1']:.3f} R@3={metrics['recall@3']:.3f} R@5={metrics['recall@5']:.3f} "
                      f"MRR={metrics['mrr@10']:.3f}")

    best = max(grid, key=lambda r: (round(r["mrr@10"], 4), r["recall@5"], -r["chunks"]))
    print(f"\nBest configuration: {best['embedding_model']} {best['strategy']} size={best['chunk_size']} "
          f"overlap={best['overlap']} {best['mode']}")

    embedder = embedders[best["embedding_model"]]
    retriever, _, _ = build_index(documents, embedder, best["strategy"], best["chunk_size"], best["overlap"])
    calibration = calibrate_threshold(retriever, examples, best["mode"], ExtractiveGenerator(embedder), settings.top_k)
    threshold = calibration["sentence"]["best"]["threshold"]
    for name, cal in calibration.items():
        print(f"{name:>8}-level gate: AUC {cal['auc']:.3f}, best threshold {cal['best']['threshold']:.2f} "
              f"(balanced accuracy {cal['best']['balanced_accuracy']:.3f})")

    run_settings = settings.model_copy(update={
        "embedding_model": best["embedding_model"], "chunk_size": best["chunk_size"], "chunk_overlap": best["overlap"],
        "retrieval_mode": best["mode"], "evidence_threshold": threshold, "chunk_strategy": best["strategy"],
    })
    end_to_end = []
    generators = [None]
    llm = build_provider(run_settings)
    if llm is not None:
        generators.append(llm)
    for provider in generators:
        store = VectorStore(None, collection_name=f"e2e-{time.time_ns()}")
        pipeline = PolicyPilot(run_settings, embedder=embedder, store=store, llm=provider)
        for doc in documents:
            pipeline.ingest(doc, refresh=False)
        pipeline.retriever.refresh()
        result = evaluate_answers(pipeline, examples, embedders[EMBEDDING_MODELS[0]])
        end_to_end.append(result)
        s = result["summary"]
        print(f"[{result['provider']}] answer rate {s['answer_rate']:.3f}, correct refusals {s['correct_refusal_rate']:.3f}, "
              f"fact recall {s['fact_recall']:.3f}, citation acc {s['citation_accuracy']:.3f}, "
              f"sentence support {s['sentence_support']:.3f}")

    results = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "environment": {"python": platform.python_version(), "machine": platform.machine(), "processor": platform.processor() or "unknown"},
        "dataset": {"questions": len(examples), "answerable": len(answerable), "unanswerable": len(examples) - len(answerable), "documents": len(documents)},
        "retrieval_grid": grid,
        "best_config": {k: best[k] for k in ("embedding_model", "strategy", "chunk_size", "overlap", "mode")},
        "threshold_calibration": calibration,
        "end_to_end": end_to_end,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "latest.json").write_text(json.dumps(results, indent=2))
    write_report(results, args.out / "report.md")
    print(f"\nWrote {args.out / 'latest.json'} and {args.out / 'report.md'}")


if __name__ == "__main__":
    main()
