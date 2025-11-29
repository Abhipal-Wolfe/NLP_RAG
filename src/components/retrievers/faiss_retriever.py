"""FAISS-based retriever implementation"""

from typing import Dict, List, Optional
import faiss
import json
import numpy as np
from ...core.document import Document

from ...core.interfaces import Retriever, RetrievalResult
from ...core.registry import register_component


@register_component("retriever", "FAISSRetriever")
class FAISSRetriever(Retriever):
    """
    FAISS-based retriever with multi-corpus support.

    Retrieves documents from multiple FAISS indices (PubMed, PMC, CPG, Textbook)
    and reranks globally by similarity score.
    """

    def __init__(
        self,
        faiss_index_paths: Dict[str, str],
        articles_paths: Dict[str, str],
        embedder: Optional[any] = None
    ):
        """
        Args:
            faiss_index_paths: {"pubmed": "path/to/index.faiss", ...}
            articles_paths: {"pubmed": "path/to/articles.json", ...}
            embedder: Optional embedding model for query encoding
        """
        self.indices = {k: faiss.read_index(v) for k, v in faiss_index_paths.items()}
        self.articles = {
            k: json.load(open(v, 'r', encoding='utf-8'))
            for k, v in articles_paths.items()
        }
        self.embedder = embedder

    def retrieve(self, query: str, k: int = 5, query_embedding: Optional[np.ndarray] = None, **kwargs) -> RetrievalResult:
        """
        Retrieve top-k documents using FAISS search.

        Args:
            query: Query string (used if query_embedding not provided)
            k: Number of documents to retrieve
            query_embedding: Pre-computed query embedding (optional)
            **kwargs: Additional arguments

        Returns:
            RetrievalResult with documents and scores
        """
        # Get query embedding
        if query_embedding is None:
            if self.embedder is None:
                raise ValueError("Either query_embedding or embedder must be provided")
            query_embedding = self.embedder.encode([query])

        # Ensure correct shape and dtype
        if len(query_embedding.shape) == 1:
            query_embedding = query_embedding.reshape(1, -1)
        query_embedding = query_embedding.astype(np.float32)

        all_results = []

        # Search each corpus
        for corpus, index in self.indices.items():
            D, I = index.search(query_embedding, k)  # distances and indices
            articles_list = self.articles[corpus]
            articles_len = len(articles_list)

            # Map indices to articles
            for distances_row, indices_row in zip(D, I):
                for dist, idx in zip(distances_row, indices_row):
                    # Skip invalid indices
                    if idx >= articles_len or idx < 0:
                        continue

                    # Extract document text
                    doc_text = (
                        articles_list[idx]
                        if isinstance(articles_list[idx], str)
                        else articles_list[idx].get("text", "")
                    )

                    all_results.append((
                        float(dist),
                        Document(
                            page_content=doc_text,
                            metadata={
                                "corpus": corpus,
                                "doc_id": int(idx),
                                "score": float(dist)
                            }
                        )
                    ))

        # Rerank top-k globally by score (higher is better for FAISS inner product)
        all_results.sort(key=lambda x: x[0], reverse=True)
        top_results = all_results[:k]

        if top_results:
            scores, docs = zip(*top_results)
        else:
            scores, docs = [], []

        return RetrievalResult(
            documents=list(docs),
            scores=list(scores),
            metadata={
                "num_candidates": len(all_results),
                "corpora": list(self.indices.keys())
            }
        )
