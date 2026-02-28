import os
import glob
import json
import hashlib
from datetime import datetime
from typing import Dict, List, Tuple

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma  # ✅ new import (clean, no deprecation)

from app.config import load_config


# ---------------------------
# Helpers: hashing + state
# ---------------------------

def sha256_file(path: str) -> str:
    """
    Returns SHA256 hash of a file.
    Why:
    - if file content changes, hash changes
    - we can detect changed files and re-index only those
    """
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def state_path(persist_dir: str) -> str:
    """
    State file lives alongside the vector DB.
    """
    return os.path.join(persist_dir, "index_state.json")


def load_state(persist_dir: str) -> Dict[str, str]:
    """
    Loads {file_path: file_hash} from disk.
    If not present, return empty dict (first run).
    """
    sp = state_path(persist_dir)
    if not os.path.exists(sp):
        return {}
    with open(sp, "r", encoding="utf-8") as f:
        return json.load(f)


def save_state(persist_dir: str, state: Dict[str, str]) -> None:
    """
    Saves {file_path: file_hash} to disk.
    """
    os.makedirs(persist_dir, exist_ok=True)
    sp = state_path(persist_dir)
    with open(sp, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


# ---------------------------
# Loading PDFs
# ---------------------------

def load_pdf_pages(pdf_path: str):
    """
    Loads ONE PDF and returns a list of Documents (1 per page).
    """
    loader = PyPDFLoader(pdf_path)
    return loader.load()


def get_pdf_paths(data_dir: str) -> List[str]:
    """
    Finds all PDFs in data/ folder.
    glob returns a list of matching paths.
    """
    return glob.glob(os.path.join(data_dir, "*.pdf"))


def decide_which_files_to_index(pdf_paths: List[str], old_state: Dict[str, str]) -> Tuple[List[str], Dict[str, str]]:
    """
    Compares current file hashes with previous state and decides which files to index.
    Also deduplicates files that have identical content (same hash).

    - If file is new or changed -> candidate for indexing
    - If another file with same hash already selected -> skip (duplicate content)
    """
    new_state = dict(old_state)
    to_index = []
    seen_hashes = set()

    for path in pdf_paths:
        current_hash = sha256_file(path)
        previous_hash = old_state.get(path)

        # Always update state with the latest hash
        new_state[path] = current_hash

        # If unchanged, skip
        if previous_hash == current_hash:
            continue

        # If duplicate content, skip indexing this file
        if current_hash in seen_hashes:
            print(f"⚠️ Skipping duplicate-content PDF: {path}")
            continue

        seen_hashes.add(current_hash)
        to_index.append(path)

    return to_index, new_state


# ---------------------------
# Main ingestion
# ---------------------------

def main():
    cfg = load_config()

    # 1) Load previous indexing state (hashes)
    old_state = load_state(cfg.vectorstore.persist_dir)

    # 2) Find PDFs and decide what needs indexing
    pdf_paths = get_pdf_paths("data")
    to_index, new_state = decide_which_files_to_index(pdf_paths, old_state)

    if not pdf_paths:
        print("No PDFs found in data/. Add PDFs and run again.")
        return

    if not to_index:
        print("✅ No changes detected. Skipping ingestion (everything already indexed).")
        return

    print("PDFs detected:", len(pdf_paths))
    print("PDFs to (re)index:", len(to_index))
    for p in to_index:
        print(" -", p)

    # 3) Load ONLY changed/new PDFs into Documents
    docs = []
    for path in to_index:
        docs.extend(load_pdf_pages(path))

    print(f"Loaded {len(docs)} pages from changed PDFs.")

    # 4) Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=cfg.chunking.chunk_size,
        chunk_overlap=cfg.chunking.chunk_overlap,
    )
    chunks = splitter.split_documents(docs)
    print(f"Created {len(chunks)} chunks from changed PDFs.")

    # 5) Embeddings (free local model)
    embeddings = HuggingFaceEmbeddings(model_name=cfg.embedding.model)

    # 6) Open existing Chroma DB
    vectordb = Chroma(
        collection_name=cfg.vectorstore.collection,
        persist_directory=cfg.vectorstore.persist_dir,
        embedding_function=embeddings,
    )

    # 7) DELETE old chunks for changed PDFs (IMPORTANT)
    for pdf_path in to_index:
        vectordb.delete(where={"source": pdf_path})
        print(f"🗑️ Deleted old chunks for: {pdf_path}")

    # 8) Create stable IDs for each chunk
    chunk_ids = []
    for idx, doc in enumerate(chunks):
        src = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "na")
        chunk_ids.append(f"{src}::page_{page}::chunk_{idx}")

    # 9) Add new chunks with explicit IDs
    vectordb.add_documents(chunks, ids=chunk_ids)
    print(f"✅ Added {len(chunks)} fresh chunks to vector DB.")

    # 7) Save updated state (hashes)
    save_state(cfg.vectorstore.persist_dir, new_state)
    print(f"✅ Updated index state at: {state_path(cfg.vectorstore.persist_dir)}")
    print("Done at:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


if __name__ == "__main__":
    main()