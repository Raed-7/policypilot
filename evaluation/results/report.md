# Evaluation report

Generated 2026-11-27T00:54:59+00:00 on Python 3.13.18 (x86_64, CPU).

Dataset: 64 hand-labelled questions (54 answerable, 10 unanswerable) over 6 policy documents.

## Retrieval

Recall@k is the share of answerable questions whose labelled evidence appears in the top k chunks; MRR@10 is the mean reciprocal rank of the first relevant chunk.

| Embedding model | Chunking | Size | Overlap | Mode | Chunks | Recall@1 | Recall@3 | Recall@5 | MRR@10 |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|
| all-MiniLM-L6-v2 **(selected)** | fixed | 256 | 64 | hybrid | 19 | 94.4% | 98.1% | 98.1% | 0.966 |
| bge-small-en-v1.5 | fixed | 256 | 0 | hybrid | 16 | 88.9% | 98.1% | 98.1% | 0.937 |
| all-MiniLM-L6-v2 | section | 128 | 0 | hybrid | 63 | 90.7% | 96.3% | 96.3% | 0.937 |
| all-MiniLM-L6-v2 | section | 128 | 32 | hybrid | 63 | 90.7% | 96.3% | 96.3% | 0.937 |
| all-MiniLM-L6-v2 | section | 256 | 0 | hybrid | 62 | 90.7% | 96.3% | 96.3% | 0.937 |
| all-MiniLM-L6-v2 | section | 256 | 64 | hybrid | 62 | 90.7% | 96.3% | 96.3% | 0.937 |
| bge-small-en-v1.5 | fixed | 256 | 64 | hybrid | 19 | 88.9% | 98.1% | 98.1% | 0.935 |
| bge-small-en-v1.5 | fixed | 128 | 32 | hybrid | 36 | 87.0% | 98.1% | 98.1% | 0.926 |
| all-MiniLM-L6-v2 | section | 64 | 16 | hybrid | 78 | 88.9% | 96.3% | 96.3% | 0.925 |
| all-MiniLM-L6-v2 | fixed | 256 | 0 | hybrid | 16 | 87.0% | 94.4% | 100.0% | 0.921 |
| bge-small-en-v1.5 | section | 128 | 0 | hybrid | 63 | 87.0% | 96.3% | 96.3% | 0.919 |
| bge-small-en-v1.5 | section | 128 | 32 | hybrid | 63 | 87.0% | 96.3% | 96.3% | 0.919 |
| bge-small-en-v1.5 | section | 256 | 0 | hybrid | 62 | 87.0% | 96.3% | 96.3% | 0.919 |
| bge-small-en-v1.5 | section | 256 | 64 | hybrid | 62 | 87.0% | 96.3% | 96.3% | 0.919 |
| all-MiniLM-L6-v2 | fixed | 256 | 64 | dense | 19 | 85.2% | 98.1% | 100.0% | 0.918 |
| bge-small-en-v1.5 | section | 64 | 0 | dense | 78 | 85.2% | 96.3% | 96.3% | 0.914 |
| all-MiniLM-L6-v2 | fixed | 128 | 32 | hybrid | 36 | 83.3% | 100.0% | 100.0% | 0.907 |
| bge-small-en-v1.5 | section | 64 | 16 | hybrid | 78 | 83.3% | 96.3% | 96.3% | 0.904 |
| all-MiniLM-L6-v2 | section | 64 | 0 | hybrid | 78 | 87.0% | 94.4% | 94.4% | 0.904 |
| bge-small-en-v1.5 | fixed | 128 | 0 | hybrid | 34 | 83.3% | 96.3% | 98.1% | 0.904 |
| bge-small-en-v1.5 | section | 64 | 0 | hybrid | 78 | 85.2% | 94.4% | 94.4% | 0.904 |
| all-MiniLM-L6-v2 | fixed | 64 | 16 | hybrid | 70 | 83.3% | 96.3% | 98.1% | 0.900 |
| bge-small-en-v1.5 | section | 128 | 0 | dense | 63 | 83.3% | 94.4% | 94.4% | 0.894 |
| bge-small-en-v1.5 | section | 256 | 0 | dense | 62 | 83.3% | 94.4% | 94.4% | 0.894 |
| bge-small-en-v1.5 | section | 256 | 64 | dense | 62 | 83.3% | 94.4% | 94.4% | 0.894 |
| all-MiniLM-L6-v2 | fixed | 128 | 0 | hybrid | 34 | 81.5% | 96.3% | 98.1% | 0.892 |
| all-MiniLM-L6-v2 | section | 64 | 0 | dense | 78 | 83.3% | 92.6% | 94.4% | 0.892 |
| bge-small-en-v1.5 | section | 128 | 32 | dense | 63 | 83.3% | 94.4% | 94.4% | 0.891 |
| bge-small-en-v1.5 | fixed | 64 | 0 | dense | 61 | 81.5% | 94.4% | 96.3% | 0.887 |
| all-MiniLM-L6-v2 | fixed | 64 | 0 | hybrid | 61 | 81.5% | 96.3% | 98.1% | 0.886 |
| all-MiniLM-L6-v2 | section | 128 | 0 | dense | 63 | 81.5% | 94.4% | 96.3% | 0.884 |
| all-MiniLM-L6-v2 | section | 256 | 0 | dense | 62 | 81.5% | 94.4% | 96.3% | 0.884 |
| all-MiniLM-L6-v2 | section | 256 | 64 | dense | 62 | 81.5% | 94.4% | 96.3% | 0.884 |
| all-MiniLM-L6-v2 | section | 128 | 32 | dense | 63 | 81.5% | 94.4% | 96.3% | 0.881 |
| bge-small-en-v1.5 | section | 64 | 16 | dense | 78 | 79.6% | 96.3% | 96.3% | 0.879 |
| bge-small-en-v1.5 | fixed | 128 | 32 | dense | 36 | 77.8% | 96.3% | 98.1% | 0.877 |
| bge-small-en-v1.5 | fixed | 64 | 16 | hybrid | 70 | 77.8% | 96.3% | 98.1% | 0.873 |
| bge-small-en-v1.5 | fixed | 64 | 0 | hybrid | 61 | 77.8% | 98.1% | 98.1% | 0.872 |
| bge-small-en-v1.5 | fixed | 256 | 0 | dense | 16 | 77.8% | 96.3% | 98.1% | 0.872 |
| all-MiniLM-L6-v2 | fixed | 128 | 32 | dense | 36 | 77.8% | 96.3% | 98.1% | 0.872 |
| all-MiniLM-L6-v2 | section | 64 | 16 | dense | 78 | 79.6% | 92.6% | 94.4% | 0.864 |
| all-MiniLM-L6-v2 | fixed | 128 | 0 | dense | 34 | 75.9% | 94.4% | 100.0% | 0.862 |
| all-MiniLM-L6-v2 | fixed | 256 | 0 | dense | 16 | 75.9% | 96.3% | 96.3% | 0.857 |
| all-MiniLM-L6-v2 | fixed | 64 | 16 | dense | 70 | 79.6% | 90.7% | 92.6% | 0.856 |
| all-MiniLM-L6-v2 | fixed | 64 | 0 | dense | 61 | 75.9% | 94.4% | 98.1% | 0.854 |
| bge-small-en-v1.5 | fixed | 256 | 64 | dense | 19 | 74.1% | 94.4% | 96.3% | 0.852 |
| bge-small-en-v1.5 | fixed | 128 | 0 | dense | 34 | 74.1% | 92.6% | 98.1% | 0.843 |
| bge-small-en-v1.5 | fixed | 64 | 16 | dense | 70 | 70.4% | 92.6% | 96.3% | 0.820 |

## Refusal gate

Questions are only answered when the retrieved text contains strong enough evidence. Two answerability signals were compared on all questions (answerable vs. unanswerable):

| Signal | ROC AUC | Best threshold | Answers answerable | Refuses unanswerable | Balanced accuracy |
|---|---:|---:|---:|---:|---:|
| Top chunk similarity | 0.883 | 0.37 | 79.6% | 90.0% | 84.8% |
| Best sentence similarity **(used)** | 0.930 | 0.56 | 77.8% | 100.0% | 88.9% |

The threshold is tuned on the same small question set it is reported on, so treat these figures as optimistic for unseen documents.

## End-to-end answers

| Generator | Answer rate | Correct refusals | Fact recall | Citation accuracy | Sentence support | Token precision | p50 latency | p95 latency |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| extractive (sentence-selection) | 75.9% | 100.0% | 86.6% | 100.0% | 100.0% | 100.0% | 136 ms | 193 ms |

- **Answer rate**: answerable questions that received an answer rather than a refusal.
- **Correct refusals**: unanswerable questions the system declined to answer.
- **Fact recall**: share of labelled key facts (figures, deadlines, names) present in answered responses.
- **Citation accuracy**: answered responses citing the labelled source document and page.
- **Sentence support**: answer sentences whose closest source sentence has cosine similarity of at least 0.75.
- **Token precision**: answer content words that occur in the retrieved excerpts.
