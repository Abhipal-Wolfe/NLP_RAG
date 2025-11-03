import os
import numpy as np
from retriever_models import FaissJsonRetriever
from embed_queries import query_preprocess, query_encode

# -----------------------------
# Paths to FAISS indices and articles
# -----------------------------

faiss_index_paths = {
    "pubmed": "path_to_pubmed_index.faiss",
    "pmc": "path_to_pmc_index.faiss",
    "cpg": "path_to_cpg_index.faiss",
    "textbook": "path_to_textbook_index.faiss",
}

articles_paths = {
    "pubmed": "path_to_pubmed_articles.json",
    "pmc":  "path_to_pmc_articles.json",
    "cpg": "path_to_cpg_articles.json",
    "textbook": "path_to_textbook_articles.json",
}

# -----------------------------
# Load retriever
# -----------------------------
retriever = FaissJsonRetriever(faiss_index_paths, articles_paths)

# -----------------------------
# Load queries
# -----------------------------
input_json_path = os.path.join("path_to_input_queries.json")
queries = query_preprocess(input_json_path, use_spacy=False)  # or True if you want [SEP] handling
query_embeddings = query_encode(queries)  # returns np.array of shape (num_queries, 768)

# -----------------------------
# Retrieve top-k for each query
# -----------------------------
top_k = 5
for i, q_emb in enumerate(query_embeddings):
    q_emb = q_emb.reshape(1, -1)  # ensure shape (1, dim)
    result = retriever.retrieve(q_emb, k=top_k)
    print(f"\nQuery {i}: {queries[i][:80]}...")
    for j, (doc, score) in enumerate(zip(result.documents, result.scores)):
        corpus = doc.metadata.get("corpus", "")
        print(f"Rank {j+1} | Score: {score:.4f} | Corpus: {corpus}")
        print(f"Snippet: {doc.page_content[:200]}...\n")