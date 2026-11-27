from typing import Literal

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    top_k: int | None = Field(None, ge=1, le=20)
    threshold: float | None = Field(None, ge=-1.0, le=1.0)
    mode: Literal["dense", "hybrid"] | None = None


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    top_k: int | None = Field(None, ge=1, le=50)
    threshold: float | None = Field(None, ge=-1.0, le=1.0)
    mode: Literal["dense", "hybrid"] | None = None


class SourceOut(BaseModel):
    index: int
    chunk_id: str
    doc_id: str
    source: str
    title: str
    page: int | None
    heading: str
    location: str
    score: float
    text: str
    cited: bool = False


class AskResponse(BaseModel):
    question: str
    answer: str
    answerable: bool
    confidence: float
    citations: list[SourceOut]
    sources: list[SourceOut]
    provider: str
    model: str
    latency_ms: dict[str, float]
    warnings: list[str]


class SearchResponse(BaseModel):
    query: str
    results: list[SourceOut]


class DocumentOut(BaseModel):
    doc_id: str
    source: str
    title: str
    chunks: int
    pages: int | None = None


class IngestOut(BaseModel):
    doc_id: str
    source: str
    title: str
    chunks: int
    status: str


class UploadResponse(BaseModel):
    documents: list[IngestOut]
    errors: list[dict]


class HealthResponse(BaseModel):
    status: str
    version: str
    provider: str
    model: str
    embedding_model: str
    retrieval_mode: str
    documents: int
    chunks: int
