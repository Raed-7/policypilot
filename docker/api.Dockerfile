FROM python:3.12-slim

ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
ARG EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/hf-cache

WORKDIR /app

RUN pip install --index-url "${TORCH_INDEX_URL}" torch
COPY requirements.txt .
RUN pip install -r requirements.txt

RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('${EMBEDDING_MODEL}')" \
    && chmod -R a+rX /opt/hf-cache
ENV HF_HUB_OFFLINE=1 \
    EMBEDDING_MODEL=${EMBEDDING_MODEL} \
    CHROMA_DIR=/app/data/chroma

COPY policypilot ./policypilot
COPY evaluation ./evaluation
COPY data/sample_policies ./data/sample_policies

RUN useradd --create-home --uid 1000 app && mkdir -p /app/data/chroma && chown -R app:app /app/data
USER app

EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --start-period=60s --retries=5 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=4)"

CMD ["uvicorn", "policypilot.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
