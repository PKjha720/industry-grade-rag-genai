import json
import time
from typing import List, Literal, Optional, Dict, Any, Tuple
from collections import deque
from fastapi import FastAPI
from pydantic import BaseModel, Field
from statistics import mean
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

from app.config import load_config, require_groq_key
from app.logger import log_event
from app.reranker import CrossEncoderReranker

REQ_COUNT = 0

LAT_TOTAL_MS = deque(maxlen=200)
LAT_RETRIEVAL_MS = deque(maxlen=200)
LAT_RERANK_MS = deque(maxlen=200)
LAT_LLM_MS = deque(maxlen=200)
# ---------------------------
# Prompt (JSON-only output)
# ---------------------------

PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a helpful assistant.\n"
     "You MUST return a valid JSON object and nothing else.\n"
     "Use ONLY the provided context.\n"
     "If the answer is not in the context, return JSON with clause_or_topic='unknown' and meaning='I don't know based on the provided documents.'\n"
     "Citations MUST be the exact bracket labels from the context, e.g. '[data\\\\iso27001.pdf | page 2]'.\n"
     "Do NOT invent citations.\n"
     "Every item in key_points MUST include at least one citation in the citations array.\n"
     "Each citation must be one of the bracket labels that appear in the provided context.\n"
     "JSON schema:\n"
     "{{\n"
     "  \"clause_or_topic\": string,\n"
     "  \"meaning\": string,\n"
     "  \"key_points\": [ {{\"text\": string, \"citations\": [string]}} ],\n"
     "  \"sources\": [string]\n"
     "}}\n"),
    ("human",
     "Question: {question}\n\n"
     "Context:\n{context}\n\n"
     "Return ONLY JSON.")
])

FIX_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are fixing a previous JSON answer.\n"
     "Return a valid JSON object and nothing else.\n"
     "Make sure EVERY key_points item has at least one citation in citations.\n"
     "Citations must be exact bracket labels from the provided context.\n"),
    ("human",
     "Question: {question}\n\n"
     "Context:\n{context}\n\n"
     "Broken/invalid or non-compliant JSON:\n{bad_json}\n\n"
     "Return corrected JSON only.")
])


# ---------------------------
# Retrieval helpers
# ---------------------------
def p95(values):
    if not values:
        return 0
    v = sorted(values)
    idx = int(0.95 * (len(v) - 1))
    return v[idx]
def format_context(docs: List[Document]) -> str:
    """
    Context given to the LLM. We embed citations directly into context lines:
    [source | page X] <chunk text>
    """
    parts = []
    for d in docs:
        src = d.metadata.get("source", "unknown")
        page = d.metadata.get("page", "na")
        parts.append(f"[{src} | page {page}] {d.page_content}")
    return "\n\n".join(parts)


def dedupe_by_source_page(docs: List[Document]) -> List[Document]:
    """Keep only first chunk per (source, page)."""
    seen = set()
    unique = []
    for d in docs:
        key = (d.metadata.get("source"), d.metadata.get("page"))
        if key in seen:
            continue
        seen.add(key)
        unique.append(d)
    return unique


def build_bm25_retriever_from_chroma(vectordb: Chroma, k: int, allowed_sources: list[str] | None = None) -> BM25Retriever:
    raw = vectordb.get()
    docs = []
    for text, meta in zip(raw["documents"], raw["metadatas"]):
        src = meta.get("source")
        if allowed_sources is not None and src not in allowed_sources:
            continue
        docs.append(Document(page_content=text, metadata=meta))

    bm25 = BM25Retriever.from_documents(docs)
    bm25.k = k
    return bm25


def json_citation_ok(answer_text: str) -> bool:
    """
    Validate JSON and ensure every key_points item has >= 1 bracket citation.
    """
    try:
        obj = json.loads(answer_text)
    except Exception:
        return False

    key_points = obj.get("key_points", [])
    if not isinstance(key_points, list) or len(key_points) == 0:
        return False

    for kp in key_points:
        cits = kp.get("citations", [])
        if not isinstance(cits, list) or len(cits) == 0:
            return False
        if not all(isinstance(c, str) and c.startswith("[") and c.endswith("]") for c in cits):
            return False

    return True


# ---------------------------
# API schemas
# ---------------------------

FilterMode = Literal["all", "curriculum_only"]

class AskRequest(BaseModel):
    question: str = Field(..., min_length=3)
    filter_mode: FilterMode = "all"


class AskResponse(BaseModel):
    answer: Dict[str, Any]
    retrieved_sources: List[Dict[str, Any]]
    latency_ms: int
    model: str


# ---------------------------
# App + startup
# ---------------------------

app = FastAPI(title="Industry RAG API", version="0.1.0")

cfg = load_config()
require_groq_key()

# Load shared components once (fast)
embeddings = HuggingFaceEmbeddings(model_name=cfg.embedding.model)
vectordb = Chroma(
    collection_name=cfg.vectorstore.collection,
    persist_directory=cfg.vectorstore.persist_dir,
    embedding_function=embeddings,
)
llm = ChatGroq(model=cfg.llm.model, temperature=0)
reranker = CrossEncoderReranker("cross-encoder/ms-marco-MiniLM-L-6-v2")

request_count = 0
latency_history = deque(maxlen=100)


@app.middleware("http")
async def metrics_middleware(request, call_next):
    global request_count
    start = time.time()
    response = await call_next(request)
    duration = (time.time() - start) * 1000
    request_count += 1
    latency_history.append(duration)
    return response



@app.get("/metrics")
def metrics():
    return {
        "requests_total": REQ_COUNT,
        "rolling_window": len(LAT_TOTAL_MS),
        "latency_ms": {
            "avg_total": round(mean(LAT_TOTAL_MS), 2) if LAT_TOTAL_MS else 0,
            "p95_total": round(p95(list(LAT_TOTAL_MS)), 2) if LAT_TOTAL_MS else 0,
            "avg_retrieval": round(mean(LAT_RETRIEVAL_MS), 2) if LAT_RETRIEVAL_MS else 0,
            "avg_rerank": round(mean(LAT_RERANK_MS), 2) if LAT_RERANK_MS else 0,
            "avg_llm": round(mean(LAT_LLM_MS), 2) if LAT_LLM_MS else 0,
        }
    }

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    start = time.time()

    # 1) Build retrieval config
    TOP_K = cfg.retrieval.k          # final docs for LLM (4)
    CANDIDATES =  12                # pool size to rerank (tune later)

    search_kwargs = {"k": TOP_K}
    if req.filter_mode == "curriculum_only":
        search_kwargs["filter"] = {"source": "data\\dl-curriculum.pdf"}

    allowed_sources = None
    if req.filter_mode == "curriculum_only":
        allowed_sources = ["data\\dl-curriculum.pdf"]
    # 2) Vector retriever (MMR)
    vector_retriever = vectordb.as_retriever(
    search_type="mmr",
    search_kwargs={**search_kwargs, "k": CANDIDATES, "fetch_k": 25, "lambda_mult": 0.5}
)

    # 3) BM25 retriever (keyword)
    bm25 = build_bm25_retriever_from_chroma(vectordb, k=CANDIDATES, allowed_sources=allowed_sources)

    retrieval_start = time.time()
    # 4) Hybrid retrieve + dedupe + top-k
    vector_docs = vector_retriever.invoke(req.question)
    bm25_docs = bm25.invoke(req.question)
    retrieval_time = (time.time() - retrieval_start) * 1000

# Merge + dedupe first
    candidates = dedupe_by_source_page(vector_docs + bm25_docs)

# Hard filter (prevents BM25 leakage in curriculum_only mode)
    if allowed_sources is not None:
        candidates = [d for d in candidates if d.metadata.get("source") in allowed_sources]

    rerank_start = time.time()
# Always rerank in API (final boss mode)
    if len(candidates) > TOP_K:
        docs = reranker.rerank(req.question, candidates, top_k=TOP_K)
    else:
        docs = candidates

    context = format_context(docs)
    rerank_time = (time.time() - rerank_start) * 1000
    # 5) Call model (JSON-only) + retry once if needed
    chain = PROMPT | llm

    llm_start = time.time()
    answer_msg = chain.invoke({"question": req.question, "context": context})
    llm_time = (time.time() - llm_start) * 1000

    if not json_citation_ok(answer_msg.content):
        fix_chain = FIX_PROMPT | llm
        answer_msg = fix_chain.invoke({
            "question": req.question,
            "context": context,
            "bad_json": answer_msg.content
        })

    # 6) Parse JSON (if parsing fails, return raw wrapped)
    try:
        answer_obj = json.loads(answer_msg.content)
    except Exception:
        answer_obj = {
            "clause_or_topic": "unknown",
            "meaning": "Model did not return valid JSON.",
            "key_points": [],
            "sources": [],
            "raw": answer_msg.content
        }

    def dedupe_citations(answer_obj: Dict[str, Any]) -> Dict[str, Any]:
        """
        Removes duplicate citations while preserving order.
        Why: models sometimes repeat the same citation multiple times.
        """
        def uniq(items: List[str]) -> List[str]:
            seen = set()
            out = []
            for x in items:
                if x in seen:
                    continue
                seen.add(x)
                out.append(x)
            return out

        # Dedupe citations inside key_points
        kps = answer_obj.get("key_points", [])
        if isinstance(kps, list):
            for kp in kps:
                if isinstance(kp, dict) and "citations" in kp and isinstance(kp["citations"], list):
                    kp["citations"] = uniq(kp["citations"])

        # Dedupe sources list
        if "sources" in answer_obj and isinstance(answer_obj["sources"], list):
            answer_obj["sources"] = uniq(answer_obj["sources"])

        return answer_obj


    answer_obj = dedupe_citations(answer_obj)

    elapsed_ms = int((time.time() - start) * 1000)

    global REQ_COUNT
    REQ_COUNT += 1

    LAT_TOTAL_MS.append(elapsed_ms)
    LAT_RETRIEVAL_MS.append(retrieval_time)
    LAT_RERANK_MS.append(rerank_time)
    LAT_LLM_MS.append(llm_time)

    retrieved_sources = [
        {"source": d.metadata.get("source", "unknown"), "page": d.metadata.get("page", "na")}
        for d in docs
    ]

    # 7) Log
    log_event({
        "route": "/ask",
        "question": req.question,
        "filter_mode": req.filter_mode,
        "llm_model": cfg.llm.model,
        "embedding_model": cfg.embedding.model,
        "k": cfg.retrieval.k,
        "retrieved_sources": retrieved_sources,
        "latency_ms": elapsed_ms,
        "retrieval_ms": round(retrieval_time, 2),
        "rerank_ms": round(rerank_time, 2),
        "llm_ms": round(llm_time, 2),
        "reranker": "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "candidates": CANDIDATES,
    })

    return AskResponse(
        answer=answer_obj,
        retrieved_sources=retrieved_sources,
        latency_ms=elapsed_ms,
        model=cfg.llm.model,
        timings={
        "retrieval_ms": round(retrieval_time, 2),
        "rerank_ms": round(rerank_time, 2),
        "llm_ms": round(llm_time, 2),
    }
    )