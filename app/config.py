import os
import yaml
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

class ChunkingConfig(BaseModel):
    chunk_size: int = 900
    chunk_overlap: int = 150

class EmbeddingConfig(BaseModel):
    provider: str = "hf"
    model: str = "sentence-transformers/all-MiniLM-L6-v2"

class LLMConfig(BaseModel):
    provider: str = "groq"
    model: str = "llama-3.1-8b-instant"

class VectorStoreConfig(BaseModel):
    provider: str = "chroma"
    persist_dir: str = ".chroma_db"
    collection: str = "docs"

class RetrievalConfig(BaseModel):
    k : int = 4

class AppConfig(BaseModel):
    chunking : ChunkingConfig
    embedding : EmbeddingConfig
    llm : LLMConfig
    vectorstore : VectorStoreConfig
    retrieval : RetrievalConfig

def load_config(path: str = "configs/app.yaml") -> AppConfig:

    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    return AppConfig(
        chunking=raw["chunking"],
        embedding=raw["embedding"],
        llm=raw["llm"],
        vectorstore=raw["vectorstore"],
        retrieval=raw["retrieval"],
    )
def require_groq_key() -> str:
    key = os.getenv("GROQ_API_KEY")
    if not key:
        raise ValueError("GROQ_API_KEY missing in .env")
    return key