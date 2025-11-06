import os
import json
import faiss
import numpy as np
from tqdm import tqdm
import gc

# -----------------------
# Helpers
# -----------------------
def load_total_embeddings_and_mapping(source_dir):
    """Load total embeddings and mapping from a directory."""
    embeddings_path = os.path.join(source_dir, "total.npy")
    mapping_path = os.path.join(source_dir, "total.json")

    if not os.path.exists(embeddings_path) or not os.path.exists(mapping_path):
        raise FileNotFoundError(f"Missing total.npy or total.json in {source_dir}")

    embeddings = np.load(embeddings_path).astype(np.float32)
    with open(mapping_path, 'r', encoding='utf-8') as f:
        mapping = json.load(f)
    return embeddings, mapping

def find_value_by_index(mapping, index, source_dir=None, source_type=None):
    """
    Get the actual text of an article chunk by index.

    mapping: dict {chunk_index: article_id}
    source_dir: base articles directory
    source_type: 'pubmed', 'pmc', 'textbook', 'cpg'
    """
    article_id = mapping.get(str(index), None)
    if article_id and source_dir and source_type:
        article_path = os.path.join(source_dir, source_type, f"{article_id}.txt")
        if os.path.exists(article_path):
            with open(article_path, 'r', encoding='utf-8') as f:
                return f.read()
    return article_id

# -----------------------
# FAISS Index Creation
# -----------------------
def create_faiss_index(source_dir):
    embeddings, mapping = load_total_embeddings_and_mapping(source_dir)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    
    for i in tqdm(range(0, len(embeddings), 10000), desc=f"Adding embeddings from {source_dir}", unit="batch"):
        batch = embeddings[i:i+10000]
        index.add(batch)
    
    del embeddings
    gc.collect()
    return index, mapping

# -----------------------
# Retrieval
# -----------------------
def retrieve_topk(index, mapping, query_embeddings, topk=10, source_dir=None, source_type=None):
    """
    Retrieve top-k nearest neighbors for each query embedding.

    Adds tqdm progress bar for overall query processing.
    """
    D, I = index.search(query_embeddings.astype(np.float32), topk)
    results = []

    for query_indices in tqdm(I, desc=f"Retrieving top-{topk} for {source_type}", unit="query"):
        results.append([
            find_value_by_index(mapping, int(idx), source_dir=source_dir, source_type=source_type)
            for idx in query_indices
        ])

    return results

# -----------------------
# Source-specific wrappers
# -----------------------
def pubmed_index_create(pubmed_dir):
    return create_faiss_index(pubmed_dir)

def pmc_index_create(pmc_dir):
    return create_faiss_index(pmc_dir)

def textbook_index_create(textbook_dir):
    return create_faiss_index(textbook_dir)

def cpg_index_create(cpg_dir):
    return create_faiss_index(cpg_dir)

def pubmed_decode(topk_indices, mapping):
    """Decode FAISS top-k indices into actual articles for PubMed."""
    return [[find_value_by_index(mapping, idx) for idx in indices] for indices in topk_indices]

def pmc_decode(topk_indices, mapping):
    return [[find_value_by_index(mapping, idx) for idx in indices] for indices in topk_indices]

def textbook_decode(topk_indices, mapping):
    return [[find_value_by_index(mapping, idx) for idx in indices] for indices in topk_indices]

def cpg_decode(topk_indices, mapping):
    return [[find_value_by_index(mapping, idx) for idx in indices] for indices in topk_indices]