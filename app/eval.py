import time
import yaml
from statistics import mean, median, mode
from typing import Dict, Any, List, Tuple,Optional

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

from app.config import load_config, require_groq_key
import json
from app.reranker import CrossEncoderReranker


# ---- Same prompt rules as query.py (keeps behavior consistent) ----
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

def format_context(docs: List[Document]) -> str:
    parts = []
    for d in docs:
        src = d.metadata.get("source", "unknown")
        page = d.metadata.get("page", "na")
        parts.append(f"[{src} | page {page}] {d.page_content}")
    return "\n\n".join(parts)


def dedupe_by_source_page(docs: List[Document]) -> List[Document]:
    seen = set()
    unique = []
    for d in docs:
        key = (d.metadata.get("source"), d.metadata.get("page"))
        if key in seen:
            continue
        seen.add(key)
        unique.append(d)
    return unique


def _norm(p: str) -> str:
    # Normalize Windows and Linux paths to the same format
    return (p or "").replace("\\", "/")

def build_bm25_retriever_from_chroma(vectordb, k: int, allowed_sources: Optional[List[str]] = None):
    raw = vectordb.get()
    docs: List[Document] = []

    allowed = None
    if allowed_sources is not None:
        allowed = set(_norm(x) for x in allowed_sources)

    for text, meta in zip(raw.get("documents", []), raw.get("metadatas", [])):
        src = _norm(meta.get("source", ""))
        if allowed is not None and src not in allowed:
            continue
        docs.append(Document(page_content=text, metadata=meta))

    # ✅ Empty guard: if nothing matches filter, return a retriever that returns []
    if len(docs) == 0:
        
        return None

    bm25 = BM25Retriever.from_documents(docs)
    bm25.k = k
    return bm25


def citation_compliance(answer_text: str) -> bool:
    """
    Validates JSON and ensures every key_point has >= 1 citation.
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


def run_one(question: str, mode: str, cfg, vectordb: Chroma, llm: ChatGroq, reranker=None, use_rerank: bool = False) -> Dict[str, Any]:

    TOP_K = cfg.retrieval.k
    CANDIDATES = 12
    # Normalize path to forward slash (works on Windows + Linux)
    curriculum_path = "data/dl-curriculum.pdf"
    allowed_sources = None
    if mode == "curriculum_only":
        allowed_sources = [curriculum_path]
    # Build search_kwargs
    search_kwargs = {"k": cfg.retrieval.k}
    if mode == "curriculum_only":
        search_kwargs["filter"] = {"source": curriculum_path}

    # Vector retriever (MMR)
    vector_retriever = vectordb.as_retriever(
        search_type="mmr",
        search_kwargs={**search_kwargs, "fetch_k": 25, "lambda_mult": 0.5}
    )

    # Hybrid BM25 + vector
    bm25 = build_bm25_retriever_from_chroma(vectordb, k=CANDIDATES, allowed_sources=allowed_sources)

    start = time.time()
    vector_docs = vector_retriever.invoke(question)
    bm25_docs = []
    if bm25 is not None:
        bm25_docs = bm25.invoke(question)

    candidates = dedupe_by_source_page(vector_docs + bm25_docs)

    if allowed_sources is not None:
        candidates = [d for d in candidates if d.metadata.get("source") in allowed_sources]

    if use_rerank and reranker is not None and len(candidates) > TOP_K:
        docs = reranker.rerank(question, candidates, top_k=TOP_K)
    else:
        docs = candidates[:TOP_K]

    context = format_context(docs)

    chain = PROMPT | llm
    answer = chain.invoke({"question": question, "context": context})
    # If JSON/citations are not compliant, retry once with FIX_PROMPT
    if not citation_compliance(answer.content):

        fix_chain = FIX_PROMPT | llm
        answer = fix_chain.invoke({
            "question": question,
            "context": context,
            "bad_json": answer.content
        })
    
    elapsed_ms = int((time.time() - start) * 1000)

    retrieved = [(d.metadata.get("source", "unknown"), d.metadata.get("page", "na")) for d in docs]
    unique_sources = len(set(s for s, _ in retrieved))
    unique_pages = len(set(retrieved))

    return {
        "latency_ms": elapsed_ms,
        "unique_sources": unique_sources,
        "unique_pages": unique_pages,
        "retrieved": retrieved,
        "citation_ok": citation_compliance(answer.content),
        "answer_preview": answer.content[:200].replace("\n", " ") + ("..." if len(answer.content) > 200 else ""),
    }


def main():
    require_groq_key()
    cfg = load_config()

    # Load eval questions
    with open("configs/eval_questions.yaml", "r", encoding="utf-8") as f:
        eval_cfg = yaml.safe_load(f)

    questions = eval_cfg["questions"]

    embeddings = HuggingFaceEmbeddings(model_name=cfg.embedding.model)
    vectordb = Chroma(
        collection_name=cfg.vectorstore.collection,
        persist_directory=cfg.vectorstore.persist_dir,
        embedding_function=embeddings,
    )

    llm = ChatGroq(model=cfg.llm.model, temperature=0)
    reranker = CrossEncoderReranker("cross-encoder/ms-marco-MiniLM-L-6-v2")

    results = []
    print("\n=== RAG EVAL RUN ===\n")
    for item in questions:
        qid = item["id"]
        q = item["question"]
        mode = item.get("mode", "all")

        r_base = run_one(q, mode, cfg, vectordb, llm, use_rerank=False)
        r_rerank = run_one(q, mode, cfg, vectordb, llm, reranker=reranker, use_rerank=True)

        results.append((qid + "_base", r_base))
        results.append((qid + "_rerank", r_rerank))

        print(f"[{qid} | base  ] latency={r_base['latency_ms']}ms | pages={r_base['unique_pages']} | citation_ok={r_base['citation_ok']}")
        print(f"[{qid} | rerank] latency={r_rerank['latency_ms']}ms | pages={r_rerank['unique_pages']} | citation_ok={r_rerank['citation_ok']}")
        print("  base retrieved  :", r_base["retrieved"])
        print("  rerank retrieved:", r_rerank["retrieved"])
        print()

    latencies = [r["latency_ms"] for _, r in results]
    citation_rate = sum(1 for _, r in results if r["citation_ok"]) / len(results)

    print("=== SUMMARY ===")
    print("Questions:", len(results))
    print("Latency ms (mean):", int(mean(latencies)))
    print("Latency ms (median):", int(median(latencies)))
    print("Citation compliance rate:", round(citation_rate * 100, 1), "%")


if __name__ == "__main__":
    main()