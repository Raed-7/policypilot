from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    llm_provider: Literal["auto", "groq", "gemini", "ollama", "extractive"] = "auto"
    groq_api_key: str = ""
    groq_model: str = "llama-3.1-8b-instant"
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.5-flash"
    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2"
    llm_timeout: float = 60.0
    llm_temperature: float = 0.0

    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_batch_size: int = 32

    chunk_size: int = 160
    chunk_overlap: int = 32
    top_k: int = 5
    score_threshold: float = 0.45
    retrieval_mode: Literal["dense", "hybrid"] = "hybrid"

    chroma_dir: Path = ROOT_DIR / "data" / "chroma"
    collection_name: str = "policies"
    sample_docs_dir: Path = ROOT_DIR / "data" / "sample_policies"
    seed_sample_docs: bool = True
    max_upload_mb: int = 25
    results_dir: Path = ROOT_DIR / "evaluation" / "results"


@lru_cache
def get_settings() -> Settings:
    return Settings()
