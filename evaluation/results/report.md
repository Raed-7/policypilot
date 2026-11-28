# Evaluation report

Generated 2026-11-28T02:28:51+00:00 on Python 3.13.18 (x86_64, CPU).

Dataset: 70 hand-labelled questions (60 answerable, 10 unanswerable) over 7 policy documents.

## Retrieval

Recall@k is the share of answerable questions whose labelled evidence appears in the top k chunks; MRR@10 is the mean reciprocal rank of the first relevant chunk.

| Embedding model | Chunking | Size | Overlap | Mode | Reranker | Chunks | Recall@1 | Recall@3 | Recall@5 | MRR@10 |
|---|---|---:|---:|---|---|---:|---:|---:|---:|---:|
| all-MiniLM-L6-v2 **(selected)** | fixed | 256 | 64 | hybrid | none | 21 | 95.0% | 98.3% | 98.3% | 0.969 |
| bge-small-en-v1.5 | fixed | 128 | 32 | hybrid | ms-marco-MiniLM-L-6-v2 | 39 | 93.3% | 96.7% | 96.7% | 0.954 |
| all-MiniLM-L6-v2 | fixed | 128 | 32 | hybrid | ms-marco-MiniLM-L-6-v2 | 39 | 93.3% | 96.7% | 96.7% | 0.954 |
| bge-small-en-v1.5 | fixed | 128 | 32 | dense | ms-marco-MiniLM-L-6-v2 | 39 | 93.3% | 96.7% | 96.7% | 0.952 |
| all-MiniLM-L6-v2 | fixed | 128 | 32 | dense | ms-marco-MiniLM-L-6-v2 | 39 | 93.3% | 96.7% | 96.7% | 0.952 |
| bge-small-en-v1.5 | fixed | 128 | 0 | dense | ms-marco-MiniLM-L-6-v2 | 37 | 91.7% | 96.7% | 98.3% | 0.948 |
| bge-small-en-v1.5 | fixed | 128 | 0 | hybrid | ms-marco-MiniLM-L-6-v2 | 37 | 91.7% | 96.7% | 98.3% | 0.948 |
| all-MiniLM-L6-v2 | fixed | 128 | 0 | hybrid | ms-marco-MiniLM-L-6-v2 | 37 | 91.7% | 96.7% | 98.3% | 0.948 |
| all-MiniLM-L6-v2 | fixed | 128 | 0 | dense | ms-marco-MiniLM-L-6-v2 | 37 | 91.7% | 96.7% | 98.3% | 0.946 |
| all-MiniLM-L6-v2 | section | 256 | 0 | hybrid | none | 70 | 91.7% | 96.7% | 96.7% | 0.943 |
| all-MiniLM-L6-v2 | section | 256 | 64 | hybrid | none | 70 | 91.7% | 96.7% | 96.7% | 0.943 |
| all-MiniLM-L6-v2 | section | 128 | 0 | hybrid | none | 71 | 91.7% | 96.7% | 96.7% | 0.942 |
| all-MiniLM-L6-v2 | fixed | 64 | 16 | hybrid | ms-marco-MiniLM-L-6-v2 | 76 | 90.0% | 96.7% | 98.3% | 0.940 |
| bge-small-en-v1.5 | fixed | 64 | 16 | dense | ms-marco-MiniLM-L-6-v2 | 76 | 90.0% | 96.7% | 98.3% | 0.939 |
| bge-small-en-v1.5 | fixed | 64 | 16 | hybrid | ms-marco-MiniLM-L-6-v2 | 76 | 90.0% | 96.7% | 98.3% | 0.939 |
| all-MiniLM-L6-v2 | section | 128 | 0 | dense | ms-marco-MiniLM-L-6-v2 | 71 | 90.0% | 96.7% | 98.3% | 0.939 |
| all-MiniLM-L6-v2 | section | 128 | 32 | dense | ms-marco-MiniLM-L-6-v2 | 71 | 90.0% | 96.7% | 98.3% | 0.939 |
| all-MiniLM-L6-v2 | section | 256 | 0 | dense | ms-marco-MiniLM-L-6-v2 | 70 | 90.0% | 96.7% | 98.3% | 0.939 |
| all-MiniLM-L6-v2 | section | 256 | 64 | dense | ms-marco-MiniLM-L-6-v2 | 70 | 90.0% | 96.7% | 98.3% | 0.939 |
| all-MiniLM-L6-v2 | fixed | 64 | 16 | dense | ms-marco-MiniLM-L-6-v2 | 76 | 90.0% | 96.7% | 98.3% | 0.939 |
| all-MiniLM-L6-v2 | fixed | 64 | 0 | hybrid | ms-marco-MiniLM-L-6-v2 | 67 | 90.0% | 96.7% | 98.3% | 0.939 |
| bge-small-en-v1.5 | fixed | 64 | 0 | dense | ms-marco-MiniLM-L-6-v2 | 67 | 90.0% | 96.7% | 96.7% | 0.938 |
| bge-small-en-v1.5 | fixed | 64 | 0 | hybrid | ms-marco-MiniLM-L-6-v2 | 67 | 90.0% | 96.7% | 96.7% | 0.938 |
| all-MiniLM-L6-v2 | fixed | 64 | 0 | dense | ms-marco-MiniLM-L-6-v2 | 67 | 90.0% | 96.7% | 96.7% | 0.938 |
| bge-small-en-v1.5 | section | 64 | 0 | dense | ms-marco-MiniLM-L-6-v2 | 86 | 90.0% | 96.7% | 98.3% | 0.937 |
| bge-small-en-v1.5 | section | 64 | 0 | hybrid | ms-marco-MiniLM-L-6-v2 | 86 | 90.0% | 96.7% | 98.3% | 0.937 |
| all-MiniLM-L6-v2 | section | 64 | 0 | dense | ms-marco-MiniLM-L-6-v2 | 86 | 90.0% | 95.0% | 98.3% | 0.936 |
| all-MiniLM-L6-v2 | section | 64 | 0 | hybrid | ms-marco-MiniLM-L-6-v2 | 86 | 90.0% | 96.7% | 96.7% | 0.936 |
| all-MiniLM-L6-v2 | section | 128 | 32 | hybrid | none | 71 | 90.0% | 96.7% | 96.7% | 0.933 |
| bge-small-en-v1.5 | section | 128 | 0 | dense | ms-marco-MiniLM-L-6-v2 | 71 | 88.3% | 96.7% | 96.7% | 0.931 |
| bge-small-en-v1.5 | section | 128 | 32 | dense | ms-marco-MiniLM-L-6-v2 | 71 | 88.3% | 96.7% | 96.7% | 0.931 |
| bge-small-en-v1.5 | section | 256 | 0 | dense | ms-marco-MiniLM-L-6-v2 | 70 | 88.3% | 96.7% | 96.7% | 0.931 |
| bge-small-en-v1.5 | section | 256 | 64 | dense | ms-marco-MiniLM-L-6-v2 | 70 | 88.3% | 96.7% | 96.7% | 0.931 |
| bge-small-en-v1.5 | fixed | 256 | 64 | hybrid | none | 21 | 88.3% | 98.3% | 98.3% | 0.931 |
| all-MiniLM-L6-v2 | section | 64 | 16 | hybrid | none | 86 | 90.0% | 96.7% | 96.7% | 0.931 |
| bge-small-en-v1.5 | section | 128 | 0 | hybrid | ms-marco-MiniLM-L-6-v2 | 71 | 88.3% | 96.7% | 96.7% | 0.930 |
| bge-small-en-v1.5 | section | 128 | 32 | hybrid | ms-marco-MiniLM-L-6-v2 | 71 | 88.3% | 96.7% | 96.7% | 0.930 |
| bge-small-en-v1.5 | section | 256 | 0 | hybrid | ms-marco-MiniLM-L-6-v2 | 70 | 88.3% | 96.7% | 96.7% | 0.930 |
| bge-small-en-v1.5 | section | 256 | 64 | hybrid | ms-marco-MiniLM-L-6-v2 | 70 | 88.3% | 96.7% | 96.7% | 0.930 |
| all-MiniLM-L6-v2 | section | 128 | 0 | hybrid | ms-marco-MiniLM-L-6-v2 | 71 | 88.3% | 96.7% | 96.7% | 0.930 |
| all-MiniLM-L6-v2 | section | 128 | 32 | hybrid | ms-marco-MiniLM-L-6-v2 | 71 | 88.3% | 96.7% | 96.7% | 0.930 |
| all-MiniLM-L6-v2 | section | 256 | 0 | hybrid | ms-marco-MiniLM-L-6-v2 | 70 | 88.3% | 96.7% | 96.7% | 0.930 |
| all-MiniLM-L6-v2 | section | 256 | 64 | hybrid | ms-marco-MiniLM-L-6-v2 | 70 | 88.3% | 96.7% | 96.7% | 0.930 |
| bge-small-en-v1.5 | section | 64 | 16 | dense | ms-marco-MiniLM-L-6-v2 | 86 | 88.3% | 96.7% | 98.3% | 0.928 |
| bge-small-en-v1.5 | section | 64 | 16 | hybrid | ms-marco-MiniLM-L-6-v2 | 86 | 88.3% | 96.7% | 98.3% | 0.928 |
| all-MiniLM-L6-v2 | section | 64 | 16 | hybrid | ms-marco-MiniLM-L-6-v2 | 86 | 88.3% | 96.7% | 98.3% | 0.928 |
| bge-small-en-v1.5 | section | 128 | 0 | hybrid | none | 71 | 88.3% | 96.7% | 96.7% | 0.928 |
| bge-small-en-v1.5 | section | 128 | 32 | hybrid | none | 71 | 88.3% | 96.7% | 96.7% | 0.928 |
| all-MiniLM-L6-v2 | section | 64 | 16 | dense | ms-marco-MiniLM-L-6-v2 | 86 | 88.3% | 95.0% | 100.0% | 0.927 |
| bge-small-en-v1.5 | section | 256 | 0 | hybrid | none | 70 | 88.3% | 96.7% | 96.7% | 0.927 |
| bge-small-en-v1.5 | section | 256 | 64 | hybrid | none | 70 | 88.3% | 96.7% | 96.7% | 0.927 |
| bge-small-en-v1.5 | fixed | 256 | 0 | dense | ms-marco-MiniLM-L-6-v2 | 18 | 88.3% | 95.0% | 95.0% | 0.925 |
| bge-small-en-v1.5 | fixed | 256 | 0 | hybrid | ms-marco-MiniLM-L-6-v2 | 18 | 88.3% | 95.0% | 95.0% | 0.925 |
| all-MiniLM-L6-v2 | fixed | 256 | 0 | dense | ms-marco-MiniLM-L-6-v2 | 18 | 88.3% | 95.0% | 95.0% | 0.925 |
| all-MiniLM-L6-v2 | fixed | 256 | 0 | hybrid | ms-marco-MiniLM-L-6-v2 | 18 | 88.3% | 95.0% | 95.0% | 0.925 |
| all-MiniLM-L6-v2 | fixed | 256 | 64 | hybrid | bge-reranker-base | 21 | 88.3% | 95.0% | 98.3% | 0.923 |
| bge-small-en-v1.5 | fixed | 128 | 32 | hybrid | none | 39 | 86.7% | 98.3% | 98.3% | 0.922 |
| bge-small-en-v1.5 | fixed | 256 | 0 | hybrid | none | 18 | 86.7% | 96.7% | 96.7% | 0.921 |
| bge-small-en-v1.5 | section | 64 | 0 | dense | none | 86 | 86.7% | 96.7% | 96.7% | 0.919 |
| all-MiniLM-L6-v2 | fixed | 256 | 64 | dense | none | 21 | 85.0% | 98.3% | 100.0% | 0.915 |
| all-MiniLM-L6-v2 | fixed | 128 | 32 | hybrid | none | 39 | 85.0% | 98.3% | 100.0% | 0.915 |
| all-MiniLM-L6-v2 | section | 64 | 0 | hybrid | none | 86 | 88.3% | 95.0% | 95.0% | 0.914 |
| bge-small-en-v1.5 | fixed | 128 | 0 | hybrid | none | 37 | 85.0% | 96.7% | 98.3% | 0.914 |
| bge-small-en-v1.5 | section | 64 | 16 | hybrid | none | 86 | 85.0% | 96.7% | 96.7% | 0.913 |
| all-MiniLM-L6-v2 | fixed | 128 | 0 | hybrid | none | 37 | 85.0% | 96.7% | 98.3% | 0.913 |
| bge-small-en-v1.5 | section | 64 | 0 | hybrid | none | 86 | 86.7% | 95.0% | 95.0% | 0.913 |
| all-MiniLM-L6-v2 | fixed | 256 | 0 | hybrid | none | 18 | 85.0% | 95.0% | 96.7% | 0.905 |
| all-MiniLM-L6-v2 | fixed | 64 | 16 | hybrid | none | 76 | 83.3% | 96.7% | 98.3% | 0.904 |
| bge-small-en-v1.5 | fixed | 128 | 0 | dense | bge-reranker-base | 37 | 85.0% | 93.3% | 96.7% | 0.903 |
| bge-small-en-v1.5 | section | 128 | 0 | dense | none | 71 | 85.0% | 95.0% | 95.0% | 0.902 |
| bge-small-en-v1.5 | section | 256 | 0 | dense | none | 70 | 85.0% | 95.0% | 95.0% | 0.902 |
| bge-small-en-v1.5 | section | 256 | 64 | dense | none | 70 | 85.0% | 95.0% | 95.0% | 0.902 |
| all-MiniLM-L6-v2 | section | 64 | 0 | dense | none | 86 | 85.0% | 93.3% | 95.0% | 0.902 |
| bge-small-en-v1.5 | section | 128 | 32 | dense | none | 71 | 85.0% | 95.0% | 95.0% | 0.899 |
| all-MiniLM-L6-v2 | fixed | 64 | 0 | hybrid | none | 67 | 83.3% | 96.7% | 98.3% | 0.899 |
| bge-small-en-v1.5 | fixed | 256 | 64 | dense | ms-marco-MiniLM-L-6-v2 | 21 | 83.3% | 95.0% | 95.0% | 0.897 |
| bge-small-en-v1.5 | fixed | 256 | 64 | hybrid | ms-marco-MiniLM-L-6-v2 | 21 | 83.3% | 95.0% | 95.0% | 0.897 |
| all-MiniLM-L6-v2 | fixed | 256 | 64 | dense | ms-marco-MiniLM-L-6-v2 | 21 | 83.3% | 95.0% | 95.0% | 0.897 |
| all-MiniLM-L6-v2 | fixed | 256 | 64 | hybrid | ms-marco-MiniLM-L-6-v2 | 21 | 83.3% | 95.0% | 95.0% | 0.897 |
| bge-small-en-v1.5 | fixed | 64 | 16 | hybrid | none | 76 | 81.7% | 96.7% | 98.3% | 0.894 |
| bge-small-en-v1.5 | fixed | 64 | 0 | dense | none | 67 | 81.7% | 96.7% | 96.7% | 0.894 |
| all-MiniLM-L6-v2 | section | 128 | 0 | dense | none | 71 | 83.3% | 95.0% | 96.7% | 0.893 |
| all-MiniLM-L6-v2 | section | 256 | 0 | dense | none | 70 | 83.3% | 95.0% | 96.7% | 0.893 |
| all-MiniLM-L6-v2 | section | 256 | 64 | dense | none | 70 | 83.3% | 95.0% | 96.7% | 0.893 |
| all-MiniLM-L6-v2 | section | 128 | 32 | dense | none | 71 | 83.3% | 95.0% | 96.7% | 0.890 |
| bge-small-en-v1.5 | fixed | 128 | 32 | dense | none | 39 | 80.0% | 96.7% | 98.3% | 0.890 |
| bge-small-en-v1.5 | fixed | 256 | 64 | dense | none | 21 | 81.7% | 95.0% | 96.7% | 0.889 |
| bge-small-en-v1.5 | section | 64 | 16 | dense | none | 86 | 81.7% | 96.7% | 96.7% | 0.888 |
| bge-small-en-v1.5 | fixed | 64 | 0 | hybrid | none | 67 | 80.0% | 96.7% | 98.3% | 0.884 |
| all-MiniLM-L6-v2 | fixed | 128 | 32 | hybrid | bge-reranker-base | 39 | 83.3% | 90.0% | 93.3% | 0.877 |
| all-MiniLM-L6-v2 | fixed | 128 | 32 | dense | none | 39 | 78.3% | 96.7% | 100.0% | 0.877 |
| all-MiniLM-L6-v2 | section | 64 | 16 | dense | none | 86 | 81.7% | 91.7% | 95.0% | 0.876 |
| bge-small-en-v1.5 | fixed | 128 | 32 | hybrid | bge-reranker-base | 39 | 83.3% | 90.0% | 91.7% | 0.876 |
| all-MiniLM-L6-v2 | fixed | 128 | 32 | dense | bge-reranker-base | 39 | 83.3% | 90.0% | 91.7% | 0.874 |
| bge-small-en-v1.5 | fixed | 128 | 32 | dense | bge-reranker-base | 39 | 83.3% | 90.0% | 91.7% | 0.873 |
| all-MiniLM-L6-v2 | fixed | 128 | 0 | dense | none | 37 | 78.3% | 93.3% | 100.0% | 0.871 |
| all-MiniLM-L6-v2 | fixed | 64 | 0 | dense | none | 67 | 78.3% | 93.3% | 98.3% | 0.870 |
| all-MiniLM-L6-v2 | fixed | 64 | 16 | dense | none | 76 | 78.3% | 90.0% | 93.3% | 0.857 |
| bge-small-en-v1.5 | fixed | 64 | 16 | dense | none | 76 | 75.0% | 93.3% | 96.7% | 0.850 |
| bge-small-en-v1.5 | fixed | 256 | 0 | dense | none | 18 | 73.3% | 93.3% | 96.7% | 0.839 |
| all-MiniLM-L6-v2 | fixed | 256 | 0 | dense | none | 18 | 75.0% | 95.0% | 95.0% | 0.838 |
| bge-small-en-v1.5 | fixed | 128 | 0 | dense | none | 37 | 73.3% | 93.3% | 98.3% | 0.837 |

## Refusal gate

Questions are only answered when the retrieved text contains strong enough evidence. Two answerability signals were compared on all questions (answerable vs. unanswerable):

| Signal | ROC AUC | Best threshold | Answers answerable | Refuses unanswerable | Balanced accuracy |
|---|---:|---:|---:|---:|---:|
| Top chunk similarity | 0.883 | 0.40 | 76.7% | 90.0% | 83.3% |
| Best sentence similarity **(used)** | 0.937 | 0.54 | 81.7% | 100.0% | 90.8% |

The threshold is tuned on the same small question set it is reported on, so treat these figures as optimistic for unseen documents.

## End-to-end answers

| Generator | Answer rate | Correct refusals | Fact recall | Citation accuracy | Sentence support | Token precision | p50 latency | p95 latency |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| extractive (sentence-selection) | 81.7% | 100.0% | 88.8% | 100.0% | 100.0% | 100.0% | 117 ms | 179 ms |

- **Answer rate**: answerable questions that received an answer rather than a refusal.
- **Correct refusals**: unanswerable questions the system declined to answer.
- **Fact recall**: share of labelled key facts (figures, deadlines, names) present in answered responses.
- **Citation accuracy**: answered responses citing the labelled source document and page.
- **Sentence support**: answer sentences whose closest source sentence has cosine similarity of at least 0.75.
- **Token precision**: answer content words that occur in the retrieved excerpts.

## Follow-up questions

10 two-turn conversations where the second question only makes sense with the first (for example "What's the maximum hotel rate in London?" followed by "And elsewhere in the UK?").

| Follow-up handling | Recall@1 | Recall@5 | Answer rate | Fact recall |
|---|---:|---:|---:|---:|
| Question on its own | 40.0% | 80.0% | 20.0% | 10.0% |
| With conversation history | 80.0% | 100.0% | 100.0% | 80.0% |
