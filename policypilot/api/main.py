import json
import logging
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from policypilot import __version__
from policypilot.api.schemas import (
    AskRequest,
    AskResponse,
    DocumentOut,
    HealthResponse,
    IngestOut,
    SearchRequest,
    SearchResponse,
    SourceOut,
    UploadResponse,
)
from policypilot.config import get_settings
from policypilot.ingestion import SUPPORTED_EXTENSIONS, UnsupportedFileType
from policypilot.models import RetrievedChunk
from policypilot.pipeline import PolicyPilot

logger = logging.getLogger("policypilot")


def _source(index: int, item: RetrievedChunk, cited: bool = False) -> SourceOut:
    c = item.chunk
    return SourceOut(
        index=index,
        chunk_id=c.chunk_id,
        doc_id=c.doc_id,
        source=c.source,
        title=c.title,
        page=c.page,
        heading=c.heading,
        location=c.location,
        score=round(item.dense_score, 4),
        text=c.text,
        cited=cited,
    )


def create_app(pipeline: PolicyPilot | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if pipeline is not None:
            app.state.pipeline = pipeline
        else:
            settings = get_settings()
            app.state.pipeline = PolicyPilot(settings)
            if settings.seed_sample_docs and app.state.pipeline.store.count() == 0 and settings.sample_docs_dir.exists():
                results = app.state.pipeline.ingest_directory(settings.sample_docs_dir)
                logger.info("Indexed %d sample documents", len(results))
        yield

    app = FastAPI(
        title="PolicyPilot API",
        version=__version__,
        description="Grounded question answering over company policy documents.",
        lifespan=lifespan,
    )
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

    def get_pipeline(request: Request) -> PolicyPilot:
        return request.app.state.pipeline

    @app.get("/health", response_model=HealthResponse)
    def health(pp: PolicyPilot = Depends(get_pipeline)):
        return HealthResponse(
            status="ok",
            version=__version__,
            provider=pp.provider_name,
            model=pp.model_name,
            embedding_model=pp.embedder.name,
            retrieval_mode=pp.settings.retrieval_mode,
            documents=len(pp.documents()),
            chunks=pp.store.count(),
        )

    @app.get("/documents", response_model=list[DocumentOut])
    def list_documents(pp: PolicyPilot = Depends(get_pipeline)):
        return pp.documents()

    @app.post("/documents", response_model=UploadResponse, status_code=201)
    async def upload_documents(files: list[UploadFile] = File(...), pp: PolicyPilot = Depends(get_pipeline)):
        limit = pp.settings.max_upload_mb * 1024 * 1024
        ingested, errors = [], []
        for upload in files:
            name = upload.filename or "upload"
            data = await upload.read()
            if len(data) > limit:
                errors.append({"source": name, "error": f"File exceeds {pp.settings.max_upload_mb} MB limit"})
                continue
            try:
                result = pp.ingest_bytes(data, name)
            except UnsupportedFileType as exc:
                errors.append({"source": name, "error": str(exc)})
                continue
            except Exception as exc:
                logger.exception("Failed to ingest %s", name)
                errors.append({"source": name, "error": f"Could not read file: {exc}"})
                continue
            if result.chunks == 0:
                errors.append({"source": name, "error": "No extractable text found (scanned PDFs are not supported)"})
                continue
            ingested.append(IngestOut(**result.__dict__))
        if not ingested and errors:
            raise HTTPException(status_code=422, detail=errors)
        return UploadResponse(documents=ingested, errors=errors)

    @app.delete("/documents/{doc_id}")
    def delete_document(doc_id: str, pp: PolicyPilot = Depends(get_pipeline)):
        removed = pp.delete_document(doc_id)
        if removed == 0:
            raise HTTPException(status_code=404, detail="Document not found")
        return {"doc_id": doc_id, "chunks_removed": removed}

    @app.post("/search", response_model=SearchResponse)
    def search(req: SearchRequest, pp: PolicyPilot = Depends(get_pipeline)):
        hits = pp.search(req.query, req.top_k, req.threshold, req.mode)
        return SearchResponse(query=req.query, results=[_source(i, h) for i, h in enumerate(hits, start=1)])

    @app.post("/ask", response_model=AskResponse)
    def ask(req: AskRequest, pp: PolicyPilot = Depends(get_pipeline)):
        answer = pp.ask(req.question, req.top_k, req.threshold, req.mode)
        cited = {c.index for c in answer.citations}
        sources = [_source(i, ctx, i in cited) for i, ctx in enumerate(answer.contexts, start=1)]
        return AskResponse(
            question=answer.question,
            answer=answer.answer,
            answerable=answer.answerable,
            confidence=answer.confidence,
            citations=[s for s in sources if s.cited],
            sources=sources,
            provider=answer.provider,
            model=answer.model,
            latency_ms=answer.latency_ms,
            warnings=answer.warnings,
        )

    @app.get("/evaluation")
    def evaluation(pp: PolicyPilot = Depends(get_pipeline)):
        path = pp.settings.results_dir / "latest.json"
        if not path.exists():
            raise HTTPException(status_code=404, detail="No evaluation results yet. Run `python -m evaluation.run`.")
        return json.loads(path.read_text())

    @app.get("/formats")
    def formats():
        return {"extensions": sorted(SUPPORTED_EXTENSIONS)}

    return app


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
app = create_app()
