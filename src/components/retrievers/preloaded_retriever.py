"""Preloaded evidence retriever for benchmark datasets"""

from typing import Dict, List, Optional
from ...core.document import Document

from ...core.interfaces import Retriever, RetrievalResult
from ...core.registry import register_component


@register_component("retriever", "PreloadedRetriever")
class PreloadedRetriever(Retriever):
    """
    Retriever for pre-fetched evidence (used in benchmarks).

    Evidence dict maps query_idx -> {
        'ctxs': [{'title': str, 'text': str}, ...]  # OR
        'evidence': [str, ...]  # simpler format
    }
    """

    def __init__(self, evidence_dict: Dict[int, Dict]):
        """
        Args:
            evidence_dict: Dictionary mapping query index to evidence
        """
        self.evidence_dict = evidence_dict

    def retrieve(self, query: str, k: int = 5, query_idx: Optional[int] = None, **kwargs) -> RetrievalResult:
        """
        Retrieve from preloaded evidence by query index.

        Args:
            query: Query string (not used, for interface compatibility)
            k: Number of documents to retrieve
            query_idx: Query index in evidence dict (required)
            **kwargs: Additional arguments

        Returns:
            RetrievalResult with documents
        """
        if query_idx is None:
            raise ValueError("query_idx required for PreloadedRetriever")

        evidence = self.evidence_dict.get(query_idx, {})
        documents = []

        # Format 1: Structured with title + text
        if "ctxs" in evidence:
            for ctx in evidence["ctxs"][:k]:
                doc_text = (
                    f"{ctx['title']}\n{ctx['text']}"
                    if ctx.get('title')
                    else ctx['text']
                )
                documents.append(Document(
                    page_content=doc_text,
                    metadata={
                        "title": ctx.get("title", ""),
                        "source": "preloaded"
                    }
                ))

        # Format 2: Plain text list
        elif "evidence" in evidence:
            for text in evidence["evidence"][:k]:
                documents.append(Document(
                    page_content=text,
                    metadata={"source": "preloaded"}
                ))

        return RetrievalResult(
            documents=documents,
            scores=None,
            metadata={
                "query_idx": query_idx,
                "available": len(documents)
            }
        )
