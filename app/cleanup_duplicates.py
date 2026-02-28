import os
import json
import hashlib
from typing import Dict, List

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from app.config import load_config


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def state_path(persist_dir: str) -> str:
    return os.path.join(persist_dir, "index_state.json")


def load_state(persist_dir: str) -> Dict[str, str]:
    sp = state_path(persist_dir)
    if not os.path.exists(sp):
        return {}
    with open(sp, "r", encoding="utf-8") as f:
        return json.load(f)


def save_state(persist_dir: str, state: Dict[str, str]) -> None:
    os.makedirs(persist_dir, exist_ok=True)
    sp = state_path(persist_dir)
    with open(sp, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


def main():
    cfg = load_config()

    # Open vector DB
    embeddings = HuggingFaceEmbeddings(model_name=cfg.embedding.model)
    vectordb = Chroma(
        collection_name=cfg.vectorstore.collection,
        persist_directory=cfg.vectorstore.persist_dir,
        embedding_function=embeddings,
    )

    # Load state of hashes
    state = load_state(cfg.vectorstore.persist_dir)

    # Build map: hash -> list of file paths
    hash_to_paths: Dict[str, List[str]] = {}
    for path in list(state.keys()):
        if os.path.exists(path):
            h = sha256_file(path)
            hash_to_paths.setdefault(h, []).append(path)
        else:
            # If file no longer exists, drop from state
            print(f"🧹 Removing missing file from state: {path}")
            state.pop(path, None)

    # For each hash with duplicates, keep one and delete others
    for h, paths in hash_to_paths.items():
        if len(paths) <= 1:
            continue

        keep = sorted(paths)[0]
        dups = [p for p in paths if p != keep]

        print(f"\nDuplicate content detected (hash={h[:10]}...):")
        print("  keep:", keep)
        for d in dups:
            print("  delete:", d)

            # Delete vectors belonging to that duplicate source
            vectordb.delete(where={"source": d})
            print(f"  🗑️ Deleted chunks for: {d}")

            # Remove from state so it doesn't appear again
            state.pop(d, None)

    save_state(cfg.vectorstore.persist_dir, state)
    print("\n✅ Cleanup complete. State updated.")


if __name__ == "__main__":
    main()