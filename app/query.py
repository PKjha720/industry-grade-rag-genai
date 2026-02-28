from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.retrievers import BM25Retriever
import time
from app.logger import log_event
import json

from app.config import load_config, require_groq_key

PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a helpful assistant.\n"
     "You MUST return a valid JSON object and nothing else.\n"
     "Use ONLY the provided context.\n"
     "If the answer is not in the context, return JSON with clause_or_topic='unknown' and meaning='I don't know based on the provided documents.'\n"
     "Citations MUST be the exact bracket labels from the context, e.g. '[data\\\\iso27001.pdf | page 2]'.\n"
     "Do NOT invent citations.\n"
     "Every key point MUST include at least one citation.\n"
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


def format_context(docs):
    #convert retrieved document chunks into a single string that we can send to the LLM.
    parts=[]
    for d in docs:
        src = d.metadata.get("source","unknown")
        page = d.metadata.get("page","na")
        parts.append(f"[{src} | page {page}] {d.page_content}")
    return "\n\n".join(parts)

def dedupe_by_source_page(docs):
    """
    Keeps only the first chunk for each (source, page).
    Why:
    - prevents repeated page chunks from polluting context
    - improves answer clarity and reduces token waste
    """
    seen = set()
    unique = []
    for d in docs:
        key = (d.metadata.get("source"), d.metadata.get("page"))
        if key in seen:
            continue
        seen.add(key)
        unique.append(d)
    return unique

def build_bm25_retriever_from_chroma(vectordb, k: int):
    """
    Builds a BM25 retriever using documents pulled from Chroma.

    Why:
    - BM25 is keyword-based (great for clause numbers, exact terms)
    - Vector search is semantic (great for meaning-based queries)
    - Hybrid = best of both worlds
    """
    raw = vectordb.get()  # returns dict with "documents" and "metadatas"
    docs = []
    for text, meta in zip(raw["documents"], raw["metadatas"]):
        # Recreate LangChain Document objects
        from langchain_core.documents import Document
        docs.append(Document(page_content=text, metadata=meta))

    bm25 = BM25Retriever.from_documents(docs)
    bm25.k = k
    return bm25
def main():
    cfg = load_config()
    require_groq_key()
    embeddings=HuggingFaceEmbeddings(model_name=cfg.embedding.model)
    vectordb=Chroma(
        collection_name=cfg.vectorstore.collection,
        persist_directory=cfg.vectorstore.persist_dir,
        embedding_function=embeddings,
    )
    use_filter=input("Filter to dl-curriculum.pdf only? (y/n)").strip().lower()

    search_kwargs= {"k": cfg.retrieval.k}
    if use_filter == "y":
        source_filter = "data\\dl-curriculum.pdf"
        search_kwargs["filter"] = {"source": source_filter}

    # MMR reduces redundancy by picking diverse chunks instead of near-duplicates
    retriever = vectordb.as_retriever(
        search_type="mmr",
        search_kwargs={**search_kwargs, "fetch_k": 20, "lambda_mult": 0.5}
)

    llm=ChatGroq(model=cfg.llm.model, temperature=0)
    question = input("Ask a question: ").strip()
    start = time.time()
    

    # --- Hybrid Retrieval ---
    # 1) Vector retriever (semantic)
    vector_docs = retriever.invoke(question)

    # 2) BM25 retriever (keyword)
    bm25 = build_bm25_retriever_from_chroma(vectordb, k=cfg.retrieval.k)
    bm25_docs = bm25.invoke(question)

    # 3) Merge results (vector + bm25)
    docs = vector_docs + bm25_docs

    # 4) Dedupe by (source, page)
    docs = dedupe_by_source_page(docs)

    # 5) Keep only top-k after merge (simple rule)
    docs = docs[:cfg.retrieval.k]

    print("\n--- Retrieved Sources ---")
    for d in docs:
        print("-",d.metadata.get("source","unknown"),"| page:", d.metadata.get("page","na"))
    context = format_context(docs)
    chain = PROMPT | llm
    answer = chain.invoke({"question": question, "context": context})

    print("\n--- Answer (JSON) ---\n")

    try:
        data = json.loads(answer.content)
        print(json.dumps(data, indent=2, ensure_ascii=False))
    except Exception:
        # If model returns invalid JSON, show raw output for debugging
        print("⚠️ Model did not return valid JSON. Raw output:\n")
        print(answer.content)
    elapsed_ms = int((time.time() - start) * 1000)

    retrieved = [
        {
            "source": d.metadata.get("source", "unknown"),
            "page": d.metadata.get("page", "na"),
        }
        for d in docs
    ]

    log_event({
        "question": question,
        "llm_model": cfg.llm.model,
        "embedding_model": cfg.embedding.model,
        "k": cfg.retrieval.k,
        "retrieved_sources": retrieved,
        "latency_ms": elapsed_ms,
    })
    print(f"\n(Logged to logs/rag_logs.jsonl, latency={elapsed_ms}ms)")
if __name__ == "__main__":
    main()