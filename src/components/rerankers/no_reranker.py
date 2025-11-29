"""Pass-through reranker (no reranking)"""

from typing import List, Optional
from ...core.document import Document

from ...core.interfaces import Reranker, RerankResult
from ...core.registry import register_component


@register_component("reranker", "NoReranker")
class NoReranker(Reranker):
    """
    Pass-through reranker that returns documents as-is.

    Useful for pipelines that don't need reranking.
    """

    def rerank(
        self,
        query: str,
        documents: List[Document],
        top_k: Optional[int] = None,
        **kwargs
    ) -> RerankResult:
        """
        Return documents without reranking.

        Args:
            query: Query string (ignored)
            documents: Documents to rerank
            top_k: Number of documents to return (None = all)
            **kwargs: Additional arguments (ignored)

        Returns:
            RerankResult with original documents
        """
        # Return original order, possibly limited by top_k
        if top_k is not None:
            documents = documents[:top_k]

        # Use original scores if available, otherwise assign uniform scores
        scores = [
            doc.metadata.get("score", 1.0) for doc in documents
        ]

        return RerankResult(
            documents=documents,
            scores=scores,
            metadata={"reranker": "NoReranker"}
        )
