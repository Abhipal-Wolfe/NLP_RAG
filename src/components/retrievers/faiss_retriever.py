"""FAISS-based retriever implementation"""

from typing import Dict, List, Optional, Union
import faiss
import json
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from ...core.document import Document

from ...core.interfaces import Retriever, RetrievalResult
from ...core.registry import register_component


@register_component("retriever", "FAISSRetriever")
class FAISSRetriever(Retriever):
    """
    FAISS-based retriever with single or multi-corpus support.

    Retrieves documents from FAISS index and optionally uses metadata.
    Supports both single corpus and multi-corpus configurations.
    
    IMPORTANT: Uses the same embedder model as build_faiss_index.py:
    "NeuML/pubmedbert-base-embeddings"
    """

    def __init__(
        self,
        faiss_index_paths: Union[str, Dict[str, str]],
        articles_paths: Union[str, Dict[str, str]],
        metadata_paths: Optional[Union[str, Dict[str, str]]] = None,
        embedder_model: str = "NeuML/pubmedbert-base-embeddings"
    ):
        """
        Args:
            faiss_index_paths: Single path string or {"corpus_name": "path/to/index.faiss", ...}
            articles_paths: Single path string or {"corpus_name": "path/to/articles.jsonl", ...}
            metadata_paths: Optional metadata files (JSONL format)
            embedder_model: Model name for query encoding (must match index building model)
        """
        print(f"Loading FAISS index...")
        # Handle single corpus or multi-corpus
        if isinstance(faiss_index_paths, str):
            self.indices = {"default": faiss.read_index(faiss_index_paths)}
            # Don't load full corpus - too large!
            self.articles_path = articles_paths
            self.metadata = {"default": self._load_jsonl(metadata_paths)} if metadata_paths else {}
        else:
            self.indices = {k: faiss.read_index(v) for k, v in faiss_index_paths.items()}
            self.articles_path = articles_paths
            self.metadata = (
                {k: self._load_jsonl(v) for k, v in metadata_paths.items()}
                if metadata_paths else {}
            )
        
        print(f"Loading embedder: {embedder_model}...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.embedder = SentenceTransformer(embedder_model, device=device)
        self.embedder.eval()
        print(f"Retriever ready. Device: {device}")
        
        # Build byte offset index for instant document access
        self._build_offset_index()
        
        # Track retrievals for cache management
        self.retrieval_count = 0

    def _load_jsonl(self, path: str) -> List[dict]:
        """Load JSONL file (one JSON object per line)"""
        data = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
        return data
    
    def _build_offset_index(self):
        """
        Build byte offset index for O(1) document access.
        Maps line number -> byte offset in file.
        Caches to disk for reuse.
        """
        from pathlib import Path
        import pickle
        
        path = self.articles_path if isinstance(self.articles_path, str) else list(self.articles_path.values())[0]
        cache_path = Path(path).with_suffix('.offset_index.pkl')
        
        # Load cached index if exists and is newer than corpus file
        if cache_path.exists():
            corpus_mtime = Path(path).stat().st_mtime
            cache_mtime = cache_path.stat().st_mtime
            if cache_mtime >= corpus_mtime:
                print(f"Loading offset index from cache...")
                with open(cache_path, 'rb') as f:
                    self.offset_index = pickle.load(f)
                print(f"Loaded {len(self.offset_index):,} offsets from cache")
                return
        
        # Build offset index
        print(f"Building byte offset index for fast document access...")
        print(f"This is a one-time operation, will be cached for future runs...")
        self.offset_index = []
        
        with open(path, 'rb') as f:
            offset = 0
            line_num = 0
            while True:
                self.offset_index.append(offset)
                line = f.readline()
                if not line:
                    break
                offset = f.tell()
                line_num += 1
                
                # Progress indicator
                if line_num % 1_000_000 == 0:
                    print(f"  Indexed {line_num:,} lines...")
        
        print(f"Built offset index: {len(self.offset_index):,} documents")
        
        # Cache to disk
        print(f"Caching offset index to {cache_path}...")
        with open(cache_path, 'wb') as f:
            pickle.dump(self.offset_index, f)
        print(f"Offset index cached successfully")
    
    def _get_document_by_index(self, corpus: str, idx: int) -> Optional[dict]:
        """
        Load a single document from corpus by line number using byte offset.
        O(1) instant random access without loading entire 24M document corpus into RAM.
        
        Args:
            corpus: Corpus name (e.g., "default")
            idx: Document index (line number, 0-indexed)
            
        Returns:
            Document dict or None
        """
        path = self.articles_path if isinstance(self.articles_path, str) else self.articles_path.get(corpus)
        if not path or idx >= len(self.offset_index):
            return None
        
        try:
            # Use byte offset for instant O(1) access
            offset = self.offset_index[idx]
            with open(path, 'r', encoding='utf-8') as f:
                f.seek(offset)
                line = f.readline().strip()
                if line:
                    return json.loads(line)
            return None
        except Exception as e:
            print(f"Error loading document {idx} from {corpus}: {e}")
            return None

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
        try:
            # Get query embedding
            if query_embedding is None:
                with torch.no_grad():
                    query_embedding = self.embedder.encode(
                        [query],
                        show_progress_bar=False,
                        convert_to_numpy=True,
                        normalize_embeddings=True  # Match index building
                    )
        except Exception as e:
            print(f"[ERROR] Failed to encode query: {e}")
            return RetrievalResult(documents=[], scores=[], metadata={"error": str(e)})

        # Ensure correct shape and dtype
        if len(query_embedding.shape) == 1:
            query_embedding = query_embedding.reshape(1, -1)
        query_embedding = query_embedding.astype(np.float32)

        all_results = []

        # Search each corpus
        for corpus, index in self.indices.items():
            try:
                D, I = index.search(query_embedding, k)  # distances and indices
            except Exception as e:
                print(f"[ERROR] FAISS search failed for corpus {corpus}: {e}")
                continue
            
            # Track retrievals
            self.retrieval_count += 1
            
            # Get metadata if available
            metadata_list = self.metadata.get(corpus, [])

            # Map indices to articles (load on-demand)
            for distances_row, indices_row in zip(D, I):
                for dist, idx in zip(distances_row, indices_row):
                    # Skip invalid indices
                    idx = int(idx)
                    if idx < 0:
                        continue

                    # Load document on-demand (avoids loading 24M docs into RAM)
                    try:
                        article = self._get_document_by_index(corpus, idx)
                        if article is None:
                            continue
                    except Exception as e:
                        print(f"[ERROR] Failed to load document {idx}: {e}")
                        continue
                    
                    # Extract document text
                    if isinstance(article, str):
                        doc_text = article
                    elif isinstance(article, dict):
                        doc_text = article.get("text", article.get("content", ""))
                    else:
                        doc_text = str(article)
                    
                    # Build metadata
                    doc_metadata = {
                        "corpus": corpus,
                        "doc_id": idx,
                        "score": float(dist)
                    }
                    
                    # Add metadata from metadata file if available
                    if idx < len(metadata_list):
                        meta_item = metadata_list[idx]
                        if isinstance(meta_item, dict):
                            doc_metadata.update(meta_item)

                    all_results.append((
                        float(dist),
                        Document(
                            page_content=doc_text,
                            metadata=doc_metadata
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
