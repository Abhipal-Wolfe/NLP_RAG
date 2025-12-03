import json
import multiprocessing
import os
from pathlib import Path

import numpy as np
import faiss
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# -------------------------------------------------------------------
# Config
# -------------------------------------------------------------------
CORPUS_PATH = Path("data/corpus/med_chunked_corpus.jsonl")

INDEX_DIR = Path("data/index")
INDEX_DIR.mkdir(parents=True, exist_ok=True)

INDEX_OUT_PATH = INDEX_DIR / "med_faiss_chunked.index"
META_OUT_PATH = INDEX_DIR / "med_faiss_chunked_meta.jsonl"

# Best free biomedical retrieval model for PubMed / MEDQA
MODEL_NAME = "NeuML/pubmedbert-base-embeddings"

# Tune these for speed vs memory
# OPTIMIZATION TIPS:
# - Batch size: Larger = faster GPU utilization but more memory (64, 128, 256, 512)
#   For 24GB+ GPU: Try 256-512. For 8-16GB: Use 128-256.
# - FAISS threads: More = faster index building (uses all CPU cores by default)
#   Set FAISS_OMP_THREADS env var to override (e.g., export FAISS_OMP_THREADS=23)
# - Accumulation batches: Accumulate N batches before adding to FAISS (faster)
#   Larger = faster but more memory (try 4-8 batches)
BATCH_SIZE = int(os.environ.get("FAISS_BUILD_BATCH_SIZE", "256"))
NUM_ACCUMULATE_BATCHES = int(os.environ.get("FAISS_ACCUMULATE_BATCHES", "4"))  # Accumulate 4 batches before adding
# Use all available CPU cores (leave 1 for system to avoid overloading)
NUM_FAISS_THREADS = int(os.environ.get("FAISS_OMP_THREADS", str(multiprocessing.cpu_count() - 2)))

# IVF-PQ parameters for compression
# With chunking, expect 3-5x more vectors than documents
NLIST = 16384  # More clusters for larger dataset
M = 64  # Sub-quantizers (768/64=12)
NBITS = 8
TRAIN_SIZE = 2_000_000  # More training samples for better quantization

# Checkpointing
CHECKPOINT_INTERVAL = 500_000
CHECKPOINT_PATH = INDEX_DIR / "checkpoint_chunked.npz"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(DEVICE)

# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------
def count_lines_fast(path: Path) -> int:
    """Fast line count: each line = one chunk."""
    if not path.exists():
        raise FileNotFoundError(path)

    count = 0
    with path.open("rb", buffering=1024 * 1024 * 16) as f:
        for chunk in iter(lambda: f.read(1024 * 1024 * 16), b""):
            count += chunk.count(b"\n")
    return count


def iter_jsonl(path: Path):
    """Stream chunks from JSONL file."""
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def estimate_memory_usage(n_vectors: int, method: str = "ivfpq") -> dict:
    """Estimate index memory usage."""
    if method == "flat":
        # 768 dims * 4 bytes per float32
        size_gb = (n_vectors * 768 * 4) / (1024**3)
        return {"method": "FlatIP", "size_gb": size_gb}
    elif method == "ivfpq":
        # M * NBITS / 8 bytes per vector
        size_gb = (n_vectors * M * NBITS / 8) / (1024**3)
        return {"method": "IVF-PQ", "size_gb": size_gb, "compression_ratio": "~30-40x"}
    return {}


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------
if __name__ == "__main__":
    if not CORPUS_PATH.exists():
        raise FileNotFoundError(f"Corpus not found: {CORPUS_PATH}")

    print(f"[INFO] Using device: {DEVICE}")
    print(f"[INFO] Batch size: {BATCH_SIZE}")
    print(f"[INFO] FAISS threads: {NUM_FAISS_THREADS}")

    # Set FAISS to use all CPU cores
    faiss.omp_set_num_threads(NUM_FAISS_THREADS)

    print(f"[INFO] Loading embedding model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME, device=DEVICE)
    model.eval()
    print("[INFO] Model loaded.")

    print(f"[INFO] Counting chunks in {CORPUS_PATH} ...")
    total_chunks = count_lines_fast(CORPUS_PATH)
    print(f"[INFO] Total chunks: {total_chunks:,}")
    
    # Estimate memory
    mem_estimate = estimate_memory_usage(total_chunks, "ivfpq")
    print(f"[INFO] Estimated {mem_estimate['method']} index size: ~{mem_estimate['size_gb']:.2f} GB")
    if "compression_ratio" in mem_estimate:
        print(f"[INFO] Compression ratio: {mem_estimate['compression_ratio']}")

    # Check for checkpoint
    start_from = 0
    if CHECKPOINT_PATH.exists():
        print(f"[INFO] Found checkpoint at {CHECKPOINT_PATH}")
        ckpt = np.load(CHECKPOINT_PATH, allow_pickle=True)
        start_from = int(ckpt.get('valid_chunks_processed', 0))  # Use valid chunks count
        print(f"[INFO] Resuming from valid chunk {start_from:,}")
        
        if INDEX_OUT_PATH.exists():
            print(f"[INFO] Loading partial index...")
            index = faiss.read_index(str(INDEX_OUT_PATH))
            dim = index.d
            is_trained = index.is_trained
        else:
            index = None
            dim = None
            is_trained = False
    else:
        index = None
        dim = None
        is_trained = False

    n_seen = start_from
    training_vectors = []

    # Handle meta file
    if start_from == 0:
        if META_OUT_PATH.exists():
            print(f"[WARN] Removing existing meta file: {META_OUT_PATH}")
            META_OUT_PATH.unlink()
        meta_fout = META_OUT_PATH.open("w", encoding="utf-8")
    else:
        # Resume: append to existing meta file
        meta_fout = META_OUT_PATH.open("a", encoding="utf-8")

    batch_texts: list[str] = []
    corpus_line_counter = 0  # Track position in corpus file
    valid_chunks_processed = 0  # Track number of valid (non-empty) chunks processed
    
    # Accumulate multiple batches before adding to FAISS (much faster)
    accumulated_embeddings = []
    accumulated_count = 0

    print("\n[INFO] Building compressed FAISS index (IVF-PQ) for CHUNKED corpus...")
    print(f"[INFO] Index params: nlist={NLIST}, m={M}, nbits={NBITS}")
    print(f"[INFO] Will train on first {TRAIN_SIZE:,} vectors")
    print(f"[INFO] Accumulating {NUM_ACCUMULATE_BATCHES} batches before adding to FAISS (faster)\n")

    for doc in tqdm(iter_jsonl(CORPUS_PATH), total=total_chunks, unit="chunks"):
        text = (doc.get("text") or "").strip()
        corpus_line_counter += 1
        
        if not text:
            # Empty chunk - skip entirely, don't write metadata or embedding
            continue
        
        # Skip already processed valid chunks (for checkpoint resume)
        if valid_chunks_processed < start_from:
            valid_chunks_processed += 1
            continue

        # Write metadata for this valid chunk
        meta = {
            "doc_id": doc.get("doc_id"),
            "source": doc.get("source"),
            "title": doc.get("title"),
            "meta": doc.get("meta", {}),  # includes chunk_index, total_chunks, etc.
        }
        meta_fout.write(json.dumps(meta, ensure_ascii=False) + "\n")

        batch_texts.append(text)
        valid_chunks_processed += 1

        if len(batch_texts) >= BATCH_SIZE:
            with torch.inference_mode():
                emb = model.encode(
                    batch_texts,
                    batch_size=BATCH_SIZE,
                    device=DEVICE,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                )

            if dim is None:
                dim = emb.shape[1]
                print(f"\n[INFO] Embedding dimension: {dim}")
                
                # Create compressed IVF-PQ index
                quantizer = faiss.IndexFlatIP(dim)
                index = faiss.IndexIVFPQ(quantizer, dim, NLIST, M, NBITS)
                print(f"[INFO] Created IndexIVFPQ for chunked corpus")

            # Phase 1: Accumulate training data
            if not is_trained:
                training_vectors.append(emb.astype('float32'))
                current_training_size = sum(v.shape[0] for v in training_vectors)
                
                if current_training_size >= TRAIN_SIZE:
                    print(f"\n[INFO] Training index with {current_training_size:,} vectors...")
                    train_data = np.vstack(training_vectors)
                    index.train(train_data)
                    print("[INFO] Training complete. Now adding vectors...")
                    
                    # Add all accumulated training vectors
                    for vec_batch in training_vectors:
                        index.add(vec_batch)
                        n_seen += vec_batch.shape[0]
                    
                    training_vectors.clear()
                    is_trained = True
                    print(f"[INFO] Added {n_seen:,} vectors. Continuing...\n")
            else:
                # Phase 2: Accumulate batches before adding (much faster)
                accumulated_embeddings.append(emb.astype("float32"))
                accumulated_count += 1
                
                # Add accumulated batches when we have enough
                if accumulated_count >= NUM_ACCUMULATE_BATCHES:
                    # Stack all accumulated embeddings and add at once
                    combined_emb = np.vstack(accumulated_embeddings)
                    index.add(combined_emb)
                    n_seen += combined_emb.shape[0]
                    accumulated_embeddings.clear()
                    accumulated_count = 0

            batch_texts.clear()

            # Checkpoint every N vectors
            if n_seen > 0 and n_seen % CHECKPOINT_INTERVAL == 0:
                print(f"\n[INFO] Checkpointing at {n_seen:,} vectors...")
                faiss.write_index(index, str(INDEX_OUT_PATH))
                np.savez(CHECKPOINT_PATH, n_seen=n_seen, is_trained=is_trained, 
                        valid_chunks_processed=valid_chunks_processed)
                meta_fout.flush()

    # Process last partial batch
    if batch_texts:
        with torch.inference_mode():
            emb = model.encode(
                batch_texts,
                batch_size=len(batch_texts),
                device=DEVICE,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )

        if dim is None:
            dim = emb.shape[1]
            print(f"\n[INFO] Embedding dimension: {dim}")
            quantizer = faiss.IndexFlatIP(dim)
            index = faiss.IndexIVFPQ(quantizer, dim, NLIST, M, NBITS)

        if not is_trained:
            training_vectors.append(emb.astype('float32'))
            if training_vectors:
                print(f"\n[INFO] Training index with {sum(v.shape[0] for v in training_vectors):,} vectors...")
                train_data = np.vstack(training_vectors)
                index.train(train_data)
                print("[INFO] Training complete.")
                for vec_batch in training_vectors:
                    index.add(vec_batch)
                    n_seen += vec_batch.shape[0]
                training_vectors.clear()
        else:
            # Add to accumulated batch
            accumulated_embeddings.append(emb.astype('float32'))
        
        batch_texts.clear()
    
    # Add any remaining accumulated embeddings after training
    if is_trained and accumulated_embeddings:
        combined_emb = np.vstack(accumulated_embeddings)
        index.add(combined_emb)
        n_seen += combined_emb.shape[0]
        print(f"[INFO] Added final {combined_emb.shape[0]:,} accumulated vectors")
        accumulated_embeddings.clear()

    meta_fout.close()

    print(f"\n[INFO] Total indexed vectors (chunks): {n_seen:,}")
    print(f"[INFO] Index memory usage: ~{(n_seen * M * NBITS / 8) / (1024**3):.2f} GB")
    print(f"[INFO] Saving FAISS index to: {INDEX_OUT_PATH}")
    faiss.write_index(index, str(INDEX_OUT_PATH))
    
    # Clean up checkpoint
    if CHECKPOINT_PATH.exists():
        CHECKPOINT_PATH.unlink()
        print("[INFO] Checkpoint removed.")
    
    print("\n" + "="*70)
    print("CHUNKED INDEX BUILD COMPLETE")
    print("="*70)
    print(f"Each original document was split into ~3-5 chunks")
    print(f"This enables retrieval from ANY part of the document")
    print(f"Trade-off: More storage, but MUCH better recall")
    print("="*70)
    print("[INFO] Done.")