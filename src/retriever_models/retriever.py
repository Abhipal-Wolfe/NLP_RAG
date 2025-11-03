"""Retriever implementations for Self-BioRAG"""

from typing import List, Dict, Optional
from abc import ABC, abstractmethod
from dataclasses import dataclass
import faiss
import json
import numpy as np
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore


@dataclass
class RetrievalResult:
    """Results from document retrieval"""
    documents: List[Document]
    scores: Optional[List[float]] = None
    metadata: Optional[Dict] = None


class BiomedicalRetriever(ABC):
    """Base retriever interface"""
    
    @abstractmethod
    def retrieve(self, query: str, k: int = 5, **kwargs) -> RetrievalResult:
        """Retrieve top-k documents for query"""
        pass


class FaissJsonRetriever(BiomedicalRetriever):
    """
    Retriever that uses FAISS index + articles JSON.
    Supports multiple corpora and reranking top-k.
    """
    
    def __init__(self, faiss_index_paths: Dict[str, str], articles_paths: Dict[str, str]):
        """
        faiss_index_paths: {"pubmed": "path/to/index.faiss", ...}
        articles_paths: {"pubmed": "path/to/articles.json", ...}
        """
        self.indices = {k: faiss.read_index(v) for k, v in faiss_index_paths.items()}
        self.articles = {k: json.load(open(v, 'r', encoding='utf-8')) for k, v in articles_paths.items()}

    def retrieve(self, query_embedding: np.ndarray, k: int = 5) -> RetrievalResult:
        """
        query_embedding: np.array of shape (1, dim) or (num_queries, dim)
        Returns top-k results across all corpora.
        """
        query_embedding = query_embedding.astype(np.float32)
        all_results = []

        # Search each corpus
        for corpus, index in self.indices.items():
            D, I = index.search(query_embedding, k)  # distances and indices
            articles_list = self.articles[corpus]
            
            # Map indices to articles
            for distances_row, indices_row in zip(D, I):
                for dist, idx in zip(distances_row, indices_row):
                    doc_text = articles_list[idx] if isinstance(articles_list[idx], str) else articles_list[idx].get("text", "")
                    all_results.append((dist, Document(page_content=doc_text, metadata={"corpus": corpus})))

        # Rerank top-k globally by score
        all_results.sort(key=lambda x: x[0], reverse=True)
        top_results = all_results[:k]
        docs, scores = zip(*top_results) if top_results else ([], [])
        return RetrievalResult(documents=list(docs), scores=list(scores), metadata={"num_candidates": len(all_results)})    
                                                                                                          


class VectorStoreRetriever(BiomedicalRetriever):
    """Semantic search using vector embeddings"""
    
    def __init__(self, vectorstore: VectorStore, search_kwargs: Optional[Dict] = None):
        self.vectorstore = vectorstore
        self.search_kwargs = search_kwargs or {}
    
    def retrieve(self, query: str, k: int = 5, **kwargs) -> RetrievalResult:
        """Retrieve via similarity search"""
        search_kwargs = {**self.search_kwargs, "k": k}
        docs_and_scores = self.vectorstore.similarity_search_with_score(query, **search_kwargs)
        
        return RetrievalResult(
            documents=[doc for doc, _ in docs_and_scores],
            scores=[score for _, score in docs_and_scores],
            metadata={"search_kwargs": search_kwargs}
        )


class PreloadedRetriever(BiomedicalRetriever):
    """
    Retriever for pre-fetched evidence (used in benchmarks).
    
    Evidence dict maps query_idx -> {
        'ctxs': [{'title': str, 'text': str}, ...]  # OR
        'evidence': [str, ...]  # simpler format
    }
    """
    
    def __init__(self, evidence_dict: Dict[int, Dict]):
        self.evidence_dict = evidence_dict
    
    def retrieve(self, query: str, k: int = 5, query_idx: Optional[int] = None, **kwargs) -> RetrievalResult:
        """Retrieve from preloaded evidence by query index"""
        if query_idx is None:
            raise ValueError("query_idx required for PreloadedRetriever")
        
        evidence = self.evidence_dict.get(query_idx, {})
        
        # Format 1: Structured with title + text
        if "ctxs" in evidence:
            documents = [
                Document(
                    page_content=f"{ctx['title']}\n{ctx['text']}" if ctx.get('title') else ctx['text'],
                    metadata={"title": ctx.get("title", ""), "source": "preloaded"}
                )
                for ctx in evidence["ctxs"][:k]
            ]
        # Format 2: Plain text list
        elif "evidence" in evidence:
            documents = [
                Document(page_content=text, metadata={"source": "preloaded"})
                for text in evidence["evidence"][:k]
            ]
        else:
            documents = []
        
        return RetrievalResult(
            documents=documents,
            metadata={"query_idx": query_idx, "available": len(documents)}
        )