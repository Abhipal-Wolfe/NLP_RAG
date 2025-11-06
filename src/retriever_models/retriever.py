"""Retriever implementations for Self-BioRAG"""

from typing import List, Dict, Optional
from abc import ABC, abstractmethod
from dataclasses import dataclass
import faiss
import json
import numpy as np
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
import os 
import glob as glob
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer
import retrieve as rt
import query_encode as qe
import rerank as rr
import gc
import live_query_encode as lqe


@dataclass
class RetrievalResult:
    """Results from document retrieval"""
    documents: List[Document]
    scores: Optional[List[float]] = None
    metadata: Optional[Dict] = None


class BiomedicalRetriever(ABC):
    """Base retriever interface"""
    
    @abstractmethod
    def retrieve(self, query, k: int = 5, **kwargs) -> RetrievalResult:
        """Retrieve top-k documents for query"""
        pass



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
    Retriever for prebuilt evidence JSONs (used in benchmarks or evaluation).

    It loads evidence automatically from a precomputed JSON file located in
    the `outputs/` directory.

    Expected JSON format:
    [
        {
            "query": str,
            "id" or "query_idx": int,
            "evidence": [str, ...] or "ctxs": [{"title": str, "text": str}, ...],
            ...
        },
        ...
    ]
    """

    def __init__(self, outputs_dir: str = "output", json_pattern: str = "*evidence*.json"):
        # Automatically find the most recent JSON file
        json_files = glob.glob(os.path.join(outputs_dir, json_pattern))
        print(outputs_dir)
        if not json_files:
            raise FileNotFoundError(f"No evidence JSON files found in {outputs_dir}")
        latest_file = max(json_files, key=os.path.getmtime)

        with open(latest_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Build evidence dict: query_idx -> entry
        self.evidence_dict = {}
        for item in data:
            qid = item.get("id", item.get("query_idx"))
            if qid is not None:
                self.evidence_dict[qid] = item

        print(f"[PreloadedRetriever] Loaded {len(self.evidence_dict)} entries from {latest_file}")

    def retrieve(self, query: str, k: int = 5, query_idx: Optional[int] = None, **kwargs) -> RetrievalResult:
        """Retrieve evidence from preloaded JSON using query index"""
        if query_idx is None:
            raise ValueError("query_idx required for PreloadedRetriever")

        evidence_entry = self.evidence_dict.get(query_idx, {})

        # Format 1: Structured title + text
        if "ctxs" in evidence_entry:
            documents = [
                Document(
                    page_content=f"{ctx.get('title', '')}\n{ctx.get('text', '')}".strip(),
                    metadata={"title": ctx.get("title", ""), "source": "preloaded"}
                )
                for ctx in evidence_entry["ctxs"][:k]
            ]
        # Format 2: Simple evidence list
        elif "evidence" in evidence_entry:
            documents = [
                Document(page_content=text, metadata={"source": "preloaded"})
                for text in evidence_entry["evidence"][:k]
            ]
        else:
            documents = []

        # Wrap all expected fields into metadata
        metadata = {
            "instruction": evidence_entry.get("instruction", ""),
            "input": evidence_entry.get("input", ""),
            "output": evidence_entry.get("output", ""),
            "metadata": evidence_entry.get("metadata", {}),
            "dataset_name": evidence_entry.get("dataset_name", ""),
            "topic": evidence_entry.get("topic", ""),
            "id": evidence_entry.get("id", evidence_entry.get("query_idx")),
            "evidence": [doc.page_content for doc in documents],  # raw text
        }

        return RetrievalResult(
            documents=documents,
            metadata=metadata
        )
    

class LiveRetriever(BiomedicalRetriever):
    """
    Retriever that performs live retrieval + reranking for a single query or batch of queries.

    Outputs a RetrievalResult compatible with PreloadedRetriever.
    """

    def __init__(
        self,
        embeddings_dir: str,
        articles_dir: str,
        reranker_type: str = "medcpt",  # 'medcpt' or 'bert'
        topk: int = 5,
        device: Optional[str] = None
    ):

        self.embeddings_dir = embeddings_dir
        self.articles_dir = articles_dir
        self.topk = topk
        self.reranker_type = reranker_type

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        # Load reranker
        if reranker_type == "medcpt":
            self.tokenizer = AutoTokenizer.from_pretrained("ncbi/MedCPT-Cross-Encoder")
            self.model = AutoModelForSequenceClassification.from_pretrained(
                "ncbi/MedCPT-Cross-Encoder"
            ).to(self.device).eval()
        elif reranker_type == "bert":
            self.model = SentenceTransformer("sentence-transformers/all-distilroberta-v1", device=self.device)

        # Load indexes once
        self.pubmed_index, self.pubmed_mapping = rt.pubmed_index_create(os.path.join(embeddings_dir, "pubmed"))
        self.pmc_index, self.pmc_mapping = rt.pmc_index_create(os.path.join(embeddings_dir, "pmc"))

        self.qe = qe
        self.encoder = lqe.query_encode(device=str(self.device))
        self.rr = rr
        self.gc = gc
        self.torch = torch

    def retrieve(self, query: str, k: Optional[int] = None, **kwargs) -> RetrievalResult:
        """Retrieve top-k documents for a single query string"""
        k = k or self.topk

        # qe.query_preprocess(query, use_spacy=False)
        # Encode query
        if self.reranker_type == "medcpt":
            q_emb = self.encoder.encode([query])[0]  # returns a numpy array
        elif self.reranker_type == "bert":
            q_emb = self.model.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0]

        # Retrieve top-k from PubMed & PMC
        pubmed_topk = self.rr.retrieve_topk(self.pubmed_index, self.pubmed_mapping, [q_emb], topk=k,
                                            source_dir=self.articles_dir, source_type='pubmed')[0]
        pmc_topk = self.rr.retrieve_topk(self.pmc_index, self.pmc_mapping, [q_emb], topk=k,
                                         source_dir=self.articles_dir, source_type='pmc')[0]

        # Combine query + evidence
        q_pairs, combined_evidence = self.rr.combine_query_evidence([query], [pubmed_topk], [pmc_topk])
        combined_evidence = combined_evidence[0]
        q_pairs = q_pairs[0]

        # Rerank
        if self.reranker_type == "medcpt":
            reranked = self.rr.rerank([q_pairs], [combined_evidence],
                                       tokenizer=self.tokenizer, model=self.model, device=self.device)[0]
        elif self.reranker_type == "bert":
            reranked = self.rr.rerank_sbert([q_pairs], [combined_evidence],
                                            sbert_model=self.model, device=self.device, topk=k)[0]

        # Wrap into RetrievalResult
        documents = [Document(page_content=txt, metadata={"source": "live"}) for txt in reranked[:k]]

        return RetrievalResult(
            documents=documents,
            metadata={
                "query": query,
                "available": len(documents)
            }
        )