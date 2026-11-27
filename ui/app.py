import html
import re
from datetime import datetime

import altair as alt
import pandas as pd
import streamlit as st

import api_client as api

st.set_page_config(page_title="PolicyPilot", page_icon="🧭", layout="wide")

BLUE, ORANGE = "#257de0", "#f47230"
SUGGESTIONS = [
    "How many days of annual leave do I get?",
    "What's the hotel limit for a London trip?",
    "Can I work from abroad for a few weeks?",
    "Who needs to approve a £1,200 expense claim?",
    "What should I do if my laptop is stolen?",
    "How much paternity leave is there?",
]
STOPWORDS = set("a an and are as at be by can do does for from has have how i if in is it me my of on or should the to was what when where which who why will with you your".split())

st.markdown(
    """
<style>
.block-container {padding-top: 2rem; max-width: 1200px;}
.pp-hero h1 {font-size: 2.1rem; margin-bottom: 0;}
.pp-hero p {color: #5b6472; margin-top: .25rem; font-size: 1.02rem;}
.cite {display:inline-block; min-width:1.35em; padding:0 .35em; margin:0 .1em; border-radius:6px;
       background:#e8f1fc; color:#1b64be; font-size:.78em; font-weight:600; text-align:center; vertical-align: super; line-height:1.5;}
.answer {font-size: 1.04rem; line-height: 1.65;}
.badges {display:flex; gap:.4rem; flex-wrap:wrap; margin:.6rem 0 .2rem;}
.badge {font-size:.76rem; padding:.15rem .55rem; border-radius:999px; background:#f1f3f6; color:#3c4350; border:1px solid #e3e6eb;}
.badge.good {background:#e9f7ec; color:#0a6b1b; border-color:#cbeed2;}
.badge.warn {background:#fff6e0; color:#8a5a00; border-color:#f8e3ad;}
.refusal {border-left:4px solid #f2a600; background:#fffaf0; padding:.75rem 1rem; border-radius:6px;}
.src {border:1px solid #e3e6eb; border-radius:10px; padding:.75rem .9rem; margin-bottom:.6rem; background:#fff;}
.src.cited {border-color:#9dc8f7; box-shadow: 0 0 0 1px #cce3fd inset;}
.src-head {display:flex; justify-content:space-between; gap:.5rem; font-size:.86rem; color:#3c4350; margin-bottom:.35rem;}
.src-head b {color:#191919;}
.src-body {font-size:.88rem; color:#333; line-height:1.55;}
.src-body mark {background:transparent; color:inherit; font-weight:600; padding:0;}
.src-body .ev {background:#fff1b8; border-radius:3px; padding:.05em .15em; box-decoration-break: clone;}
.bar {height:4px; background:#eef1f5; border-radius:4px; margin-top:.5rem;}
.bar > span {display:block; height:4px; background:#257de0; border-radius:4px;}
.doc-row {font-size:.86rem;}
.muted {color:#6b7280; font-size:.82rem;}
</style>
""",
    unsafe_allow_html=True,
)


def render_answer_text(text: str) -> str:
    escaped = html.escape(text)
    return re.sub(r"\[(\d+)\]", r'<span class="cite">\1</span>', escaped)


def highlight(text: str, query: str) -> str:
    escaped = html.escape(text)
    terms = {t for t in re.findall(r"[\w£%]+", query.lower()) if len(t) > 2 and t not in STOPWORDS}
    for term in sorted(terms, key=len, reverse=True):
        escaped = re.sub(rf"(?i)\b({re.escape(term)}\w*)", r"<mark>\1</mark>", escaped)
    return escaped


def evidence_excerpt(text: str, answer: str, query: str, limit: int = 900) -> str:
    """Show the part of a passage that the answer quotes, with quoted sentences emphasised."""
    quoted = [q.strip() for q in re.split(r"\s*\[\d+\]\s*", answer) if len(q.strip()) > 15]
    spans = [(text.find(q.rstrip(".")), q.rstrip(".")) for q in quoted]
    spans = sorted((i, q) for i, q in spans if i >= 0)
    if spans:
        start = max(0, spans[0][0] - 250)
        start = text.find(" ", start) + 1 if start else 0
        window = text[start : start + limit]
        out = highlight(window, query)
        for _, q in spans:
            esc = highlight(q, query)
            out = out.replace(esc, f'<span class="ev">{esc}</span>')
        prefix = "… " if start else ""
        suffix = " …" if start + limit < len(text) else ""
        return prefix + out + suffix
    if len(text) > limit:
        text = text[:limit].rsplit(" ", 1)[0] + " …"
    return highlight(text, query)


def source_card(src: dict, query: str, answer: str = "") -> str:
    location = f" · {html.escape(src['location'])}" if src["location"] else ""
    width = max(0, min(100, int(src["score"] * 100)))
    cls = "src cited" if src.get("cited") else "src"
    return (
        f'<div class="{cls}"><div class="src-head"><span><span class="cite">{src["index"]}</span> '
        f"<b>{html.escape(src['title'])}</b>{location}</span><span class='muted'>{html.escape(src['source'])} · similarity {src['score']:.2f}</span></div>"
        f'<div class="src-body">{evidence_excerpt(src["text"], answer, query)}</div><div class="bar"><span style="width:{width}%"></span></div></div>'
    )


def render_response(res: dict):
    if res["answerable"]:
        st.markdown(f'<div class="answer">{render_answer_text(res["answer"])}</div>', unsafe_allow_html=True)
    else:
        st.markdown(
            '<div class="refusal"><b>Not covered by your documents.</b><br>'
            "PolicyPilot couldn't find evidence for this in the indexed policies, so it won't guess. "
            "Try rephrasing, or check with the People Team.</div>",
            unsafe_allow_html=True,
        )
    badges = []
    if res["answerable"]:
        n = len(res["citations"])
        badges.append(f'<span class="badge good">✓ Grounded in {n} source{"s" if n != 1 else ""}</span>')
    else:
        badges.append('<span class="badge warn">Refused: insufficient evidence</span>')
    badges.append(f'<span class="badge">Evidence {res["confidence"]:.2f}</span>')
    badges.append(f'<span class="badge">{res["latency_ms"]["total"]:.0f} ms</span>')
    badges.append(f'<span class="badge">{html.escape(res["provider"])} · {html.escape(res["model"])}</span>')
    st.markdown(f'<div class="badges">{"".join(badges)}</div>', unsafe_allow_html=True)
    for warning in res.get("warnings", []):
        st.caption(f"⚠️ {warning}")

    sources = res["citations"] if res["answerable"] and res["citations"] else res["sources"]
    label = "Sources" if res["answerable"] else "Closest passages found"
    if sources:
        with st.expander(f"{label} ({len(sources)})", expanded=res["answerable"]):
            answer = res["answer"] if res["answerable"] else ""
            st.markdown("".join(source_card(s, res["question"], answer) for s in sources), unsafe_allow_html=True)


def conversation_markdown(messages: list[dict]) -> str:
    lines = [f"# PolicyPilot conversation\n\nExported {datetime.now():%Y-%m-%d %H:%M}\n"]
    for m in messages:
        if m["role"] == "user":
            lines.append(f"## Q: {m['content']}\n")
        else:
            res = m["response"]
            lines.append(res["answer"] + "\n")
            for c in res["citations"]:
                loc = f", {c['location']}" if c["location"] else ""
                lines.append(f"- [{c['index']}] {c['title']} ({c['source']}{loc})")
            lines.append("")
    return "\n".join(lines)


# ---------- sidebar ----------
try:
    status = api.health()
except api.APIError as exc:
    st.error(f"{exc}. Start it with `docker compose up` or `make api`.")
    st.stop()

with st.sidebar:
    st.markdown("### 🧭 PolicyPilot")
    st.caption(f"{status['documents']} documents · {status['chunks']} chunks indexed")
    llm_label = "Extractive (no API key)" if status["provider"] == "extractive" else f"{status['provider'].title()} · {status['model']}"
    st.caption(f"Answers: **{llm_label}**  \nEmbeddings: `{status['embedding_model'].split('/')[-1]}`")

    st.divider()
    st.markdown("**Add documents**")
    uploads = st.file_uploader(
        "PDF, Markdown or text", type=["pdf", "md", "markdown", "txt"], accept_multiple_files=True, label_visibility="collapsed"
    )
    if uploads and st.button("Index files", type="primary", use_container_width=True):
        with st.spinner("Reading, chunking and embedding…"):
            try:
                result = api.upload([(f.name, f.getvalue()) for f in uploads])
                for d in result["documents"]:
                    st.toast(f"{d['source']}: {d['status']} ({d['chunks']} chunks)", icon="✅")
                for e in result["errors"]:
                    st.toast(f"{e['source']}: {e['error']}", icon="⚠️")
            except api.APIError as exc:
                st.error(str(exc))
        st.rerun()

    st.divider()
    st.markdown("**Library**")
    for doc in api.documents():
        cols = st.columns([5, 1])
        pages = f"{doc['pages']} pages · " if doc.get("pages") else ""
        cols[0].markdown(f"<div class='doc-row'><b>{html.escape(doc['title'])}</b><br><span class='muted'>{pages}{doc['chunks']} chunks · {html.escape(doc['source'])}</span></div>", unsafe_allow_html=True)
        if cols[1].button("", icon=":material/delete:", key=f"del-{doc['doc_id']}", help=f"Remove {doc['source']}"):
            api.delete(doc["doc_id"])
            st.rerun()

    st.divider()
    with st.expander("Retrieval settings"):
        top_k = st.slider("Passages to retrieve", 1, 10, 5)
        mode = st.radio("Search mode", ["hybrid", "dense"], horizontal=True, help="Hybrid fuses semantic search with BM25 keyword search.")
    if st.session_state.get("messages"):
        st.download_button(
            "Export conversation", conversation_markdown(st.session_state.messages), "policypilot-conversation.md", use_container_width=True
        )
        if st.button("Clear conversation", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

# ---------- main ----------
st.markdown(
    '<div class="pp-hero"><h1>PolicyPilot</h1><p>Ask anything about your company policies. Every answer is quoted from your documents, '
    "with page-level citations, and PolicyPilot says so when the documents don't cover it.</p></div>",
    unsafe_allow_html=True,
)

tab_ask, tab_search, tab_eval = st.tabs(["💬 Ask", "🔎 Explore retrieval", "📊 Evaluation"])

with tab_ask:
    st.session_state.setdefault("messages", [])
    chat_box = st.container()
    question = st.chat_input("Ask about leave, expenses, remote work, security…")
    question = question or st.session_state.pop("pending", None)
    if not st.session_state.messages and not question:
        chat_box.markdown("**Try one of these:**")
        cols = chat_box.columns(3)
        for i, q in enumerate(SUGGESTIONS):
            if cols[i % 3].button(q, key=f"sugg-{i}", use_container_width=True):
                st.session_state.pending = q
                st.rerun()

    for m in st.session_state.messages:
        with chat_box.chat_message(m["role"], avatar="🧑" if m["role"] == "user" else "🧭"):
            if m["role"] == "user":
                st.markdown(m["content"])
            else:
                render_response(m["response"])

    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with chat_box.chat_message("user", avatar="🧑"):
            st.markdown(question)
        with chat_box.chat_message("assistant", avatar="🧭"):
            with st.spinner("Searching your policies…"):
                try:
                    response = api.ask(question, top_k=top_k, mode=mode)
                except api.APIError as exc:
                    st.error(str(exc))
                    st.stop()
            render_response(response)
        st.session_state.messages.append({"role": "assistant", "response": response})

with tab_search:
    st.caption("See exactly what the retriever returns for a query, before any answer is generated.")
    query = st.text_input("Query", placeholder="e.g. mileage rate for using my own car")
    if query:
        results = api.search(query, top_k=10, mode=mode, threshold=-1.0)["results"]
        if not results:
            st.info("No passages indexed yet.")
        else:
            df = pd.DataFrame(
                [{"#": r["index"], "Document": r["title"], "Location": r["location"], "Similarity": r["score"]} for r in results]
            )
            st.dataframe(
                df, hide_index=True, use_container_width=True,
                column_config={"Similarity": st.column_config.ProgressColumn("Similarity", min_value=0.0, max_value=1.0, format="%.3f")},
            )
            st.markdown("".join(source_card(r, query) for r in results[:5]), unsafe_allow_html=True)

with tab_eval:
    ev = api.evaluation()
    if not ev:
        st.info("No evaluation results yet. Run `make eval` to generate them.")
    else:
        e2e = ev["end_to_end"][0]["summary"]
        best = ev["best_config"]
        best_row = next(
            r for r in ev["retrieval_grid"] if all(r[k] == best[k] for k in ("embedding_model", "strategy", "chunk_size", "overlap", "mode"))
        )
        ds = ev["dataset"]
        st.caption(
            f"{ds['questions']} hand-labelled questions ({ds['answerable']} answerable, {ds['unanswerable']} deliberately unanswerable) "
            f"over {ds['documents']} documents · generated {ev['generated_at'][:10]}"
        )
        k = st.columns(5)
        k[0].metric("Recall@5", f"{best_row['recall@5']:.0%}")
        k[1].metric("MRR@10", f"{best_row['mrr@10']:.3f}")
        k[2].metric("Correct refusals", f"{e2e['correct_refusal_rate']:.0%}")
        k[3].metric("Citation accuracy", f"{e2e['citation_accuracy']:.0%}")
        k[4].metric("Fact recall", f"{e2e['fact_recall']:.0%}")
        st.caption(
            f"Selected configuration: `{best['embedding_model'].split('/')[-1]}`, {best['strategy']} chunks of {best['chunk_size']} words "
            f"with {best['overlap']} overlap, {best['mode']} retrieval."
        )

        grid = pd.DataFrame(ev["retrieval_grid"])
        grid["model"] = grid["embedding_model"].str.split("/").str[-1]
        grid["config"] = grid.apply(lambda r: f"{r['model']} · {r['strategy']} · {r['chunk_size']}/{r['overlap']} · {r['mode']}", axis=1)
        st.markdown("##### Top retrieval configurations")
        c1, c2 = st.columns(2)
        model_filter = c1.multiselect("Embedding model", sorted(grid["model"].unique()), default=sorted(grid["model"].unique()))
        mode_filter = c2.multiselect("Mode", ["dense", "hybrid"], default=["dense", "hybrid"])
        view = grid[grid["model"].isin(model_filter) & grid["mode"].isin(mode_filter)].sort_values("mrr@10", ascending=False).head(10)
        st.dataframe(
            view[["config", "chunks", "recall@1", "recall@5", "mrr@10"]],
            hide_index=True,
            use_container_width=True,
            column_config={
                "config": st.column_config.TextColumn("Model · chunking · size/overlap · mode", width="large"),
                "chunks": st.column_config.NumberColumn("Chunks"),
                "recall@1": st.column_config.NumberColumn("Recall@1", format="percent"),
                "recall@5": st.column_config.NumberColumn("Recall@5", format="percent"),
                "mrr@10": st.column_config.ProgressColumn("MRR@10", min_value=0.0, max_value=1.0, format="%.3f"),
            },
        )
        with st.expander("Full results table"):
            st.dataframe(
                grid[["model", "strategy", "chunk_size", "overlap", "mode", "chunks", "recall@1", "recall@3", "recall@5", "mrr@10"]]
                .sort_values("mrr@10", ascending=False),
                hide_index=True, use_container_width=True,
            )

        st.markdown("##### Choosing the refusal threshold")
        cal = ev["threshold_calibration"]["sentence"]
        sweep = pd.DataFrame(cal["sweep"]).melt(
            id_vars="threshold", value_vars=["answer_rate", "refusal_rate"], var_name="metric", value_name="rate"
        )
        sweep["metric"] = sweep["metric"].map({"answer_rate": "Answerable questions answered", "refusal_rate": "Unanswerable questions refused"})
        colors = alt.Scale(domain=["Answerable questions answered", "Unanswerable questions refused"], range=[BLUE, ORANGE])
        line = (
            alt.Chart(sweep)
            .mark_line(strokeWidth=2)
            .encode(
                x=alt.X("threshold:Q", title="Evidence threshold (best sentence similarity)"),
                y=alt.Y("rate:Q", title=None, axis=alt.Axis(format="%")),
                color=alt.Color("metric:N", scale=colors, legend=alt.Legend(orient="top", title=None)),
                tooltip=[alt.Tooltip("threshold:Q", format=".2f"), "metric:N", alt.Tooltip("rate:Q", format=".0%")],
            )
        )
        rule = alt.Chart(pd.DataFrame({"t": [cal["best"]["threshold"]]})).mark_rule(strokeDash=[4, 4], color="#666666").encode(x="t:Q")
        st.altair_chart((line + rule).properties(height=260), use_container_width=True)
        st.caption(
            f"Dashed line: selected threshold {cal['best']['threshold']:.2f}, balanced accuracy {cal['best']['balanced_accuracy']:.0%}. "
            f"Sentence-level evidence separates answerable from unanswerable questions better than whole-chunk similarity "
            f"(ROC AUC {cal['auc']:.2f} vs {ev['threshold_calibration']['chunk']['auc']:.2f})."
        )
