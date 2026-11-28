from pathlib import Path


def _pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def write_report(results: dict, path: Path) -> None:
    ds = results["dataset"]
    best = results["best_config"]
    cal = results["threshold_calibration"]
    lines = [
        "# Evaluation report",
        "",
        f"Generated {results['generated_at']} on Python {results['environment']['python']} ({results['environment']['machine']}, CPU).",
        "",
        f"Dataset: {ds['questions']} hand-labelled questions ({ds['answerable']} answerable, {ds['unanswerable']} unanswerable) "
        f"over {ds['documents']} policy documents.",
        "",
        "## Retrieval",
        "",
        "Recall@k is the share of answerable questions whose labelled evidence appears in the top k chunks; "
        "MRR@10 is the mean reciprocal rank of the first relevant chunk.",
        "",
        "| Embedding model | Chunking | Size | Overlap | Mode | Reranker | Chunks | Recall@1 | Recall@3 | Recall@5 | MRR@10 |",
        "|---|---|---:|---:|---|---|---:|---:|---:|---:|---:|",
    ]
    for r in sorted(results["retrieval_grid"], key=lambda r: -r["mrr@10"]):
        marker = " **(selected)**" if all(r[k] == best[k] for k in best) else ""
        lines.append(
            f"| {r['embedding_model'].split('/')[-1]}{marker} | {r['strategy']} | {r['chunk_size']} | {r['overlap']} | {r['mode']} | "
            f"{r['reranker'].split('/')[-1]} | "
            f"{r['chunks']} | {_pct(r['recall@1'])} | {_pct(r['recall@3'])} | {_pct(r['recall@5'])} | {r['mrr@10']:.3f} |"
        )
    sent, chunk = cal["sentence"], cal["chunk"]
    lines += [
        "",
        "## Refusal gate",
        "",
        "Questions are only answered when the retrieved text contains strong enough evidence. Two answerability "
        "signals were compared on all questions (answerable vs. unanswerable):",
        "",
        "| Signal | ROC AUC | Best threshold | Answers answerable | Refuses unanswerable | Balanced accuracy |",
        "|---|---:|---:|---:|---:|---:|",
        f"| Top chunk similarity | {chunk['auc']:.3f} | {chunk['best']['threshold']:.2f} | {_pct(chunk['best']['answer_rate'])} | "
        f"{_pct(chunk['best']['refusal_rate'])} | {_pct(chunk['best']['balanced_accuracy'])} |",
        f"| Best sentence similarity **(used)** | {sent['auc']:.3f} | {sent['best']['threshold']:.2f} | {_pct(sent['best']['answer_rate'])} | "
        f"{_pct(sent['best']['refusal_rate'])} | {_pct(sent['best']['balanced_accuracy'])} |",
        "",
        "The threshold is tuned on the same small question set it is reported on, so treat these figures as optimistic "
        "for unseen documents.",
        "",
        "## End-to-end answers",
        "",
        "| Generator | Answer rate | Correct refusals | Fact recall | Citation accuracy | Sentence support | Token precision | p50 latency | p95 latency |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for e in results["end_to_end"]:
        s = e["summary"]
        lines.append(
            f"| {e['provider']} ({e['model']}) | {_pct(s['answer_rate'])} | {_pct(s['correct_refusal_rate'])} | {_pct(s['fact_recall'])} | "
            f"{_pct(s['citation_accuracy'])} | {_pct(s['sentence_support'])} | {_pct(s['token_precision'])} | "
            f"{s['latency_ms_p50']:.0f} ms | {s['latency_ms_p95']:.0f} ms |"
        )
    lines += [
        "",
        "- **Answer rate**: answerable questions that received an answer rather than a refusal.",
        "- **Correct refusals**: unanswerable questions the system declined to answer.",
        "- **Fact recall**: share of labelled key facts (figures, deadlines, names) present in answered responses.",
        "- **Citation accuracy**: answered responses citing the labelled source document and page.",
        "- **Sentence support**: answer sentences whose closest source sentence has cosine similarity of at least 0.75.",
        "- **Token precision**: answer content words that occur in the retrieved excerpts.",
        "",
    ]
    if "followups" in results:
        fu = results["followups"]
        a, b = fu["without_history"], fu["with_history"]
        lines += [
            "## Follow-up questions",
            "",
            f"{len(b['rows'])} two-turn conversations where the second question only makes sense with the first "
            "(for example \"What's the maximum hotel rate in London?\" followed by \"And elsewhere in the UK?\").",
            "",
            "| Follow-up handling | Recall@1 | Recall@5 | Answer rate | Fact recall |",
            "|---|---:|---:|---:|---:|",
            f"| Question on its own | {_pct(a['recall@1'])} | {_pct(a['recall@5'])} | {_pct(a['answer_rate'])} | {_pct(a['fact_recall'])} |",
            f"| With conversation history | {_pct(b['recall@1'])} | {_pct(b['recall@5'])} | {_pct(b['answer_rate'])} | {_pct(b['fact_recall'])} |",
            "",
        ]
    path.write_text("\n".join(lines))
