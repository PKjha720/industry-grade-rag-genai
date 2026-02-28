from typing import List, Tuple
from sentence_transformers import CrossEncoder
from langchain_core.documents import Document


class CrossEncoderReranker:
    """
    Cross-encoder reranker:
    - Takes (query, doc_chunk) pairs
    - Scores each pair with a cross-encoder
    - Returns top_k most relevant docs

    Why it’s “final boss”:
    - Vector search is fast but approximate
    - Cross-encoder is slower but much more accurate
    - Best practice: retrieve many -> rerank -> send few to LLM
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, docs: List[Document], top_k: int = 4) -> List[Document]:
        if not docs:
            return []

        # Prepare (query, chunk_text) pairs for scoring
        pairs: List[Tuple[str, str]] = [(query, d.page_content) for d in docs]

        # Predict relevance scores (higher is better)
        scores = self.model.predict(pairs)

        # Sort docs by score descending
        ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)

        # Return top_k docs only
        return [d for d, _ in ranked[:top_k]]