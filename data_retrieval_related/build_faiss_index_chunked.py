# scripts/build_faiss_index_fast.py
import json
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
CORPUS_PATH = Path("../data/corpus/med_chunked_corpus.jsonl")

INDEX_DIR = Path("data/index")
INDEX_DIR.mkdir(parents=True, exist_ok=True)

INDEX_OUT_PATH = INDEX_DIR / "med_faiss_chunked.index"
META_OUT_PATH = INDEX_DIR / "med_faiss_chunked_meta.jsonl"

# Best free biomedical retrieval model for PubMed / MEDQA
MODEL_NAME = "NeuML/pubmedbert-base-embeddings"

# Tune these for speed vs memory
BATCH_SIZE = int(os.environ.get("FAISS_BUILD_BATCH_SIZE", "128"))  # Reduced for RTX 4090
NUM_FAISS_THREADS = int(os.environ.get("FAISS_OMP_THREADS", str(os.cpu_count() or 8)))

# IVF-PQ parameters for compression (reduces memory ~40x)
NLIST = 8192  # Number of clusters (increased for 24M scale)
M = 64  # Sub-quantizers (must divide 768 evenly: 768/64=12)
NBITS = 8  # Bits per sub-quantizer
TRAIN_SIZE = 1_000_000  # Vectors needed for training

# Checkpointing (resume from crashes)
CHECKPOINT_INTERVAL = 500_000  # Save progress every 500k vectors
CHECKPOINT_PATH = INDEX_DIR / "checkpoint.npz"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------
def count_lines_fast(path: Path) -> int:
    """Fast line count for progress bar: each line = one doc."""
    if not path.exists():
        raise FileNotFoundError(path)

    count = 0
    with path.open("rb", buffering=1024 * 1024 * 16) as f:
        for chunk in iter(lambda: f.read(1024 * 1024 * 16), b""):
            count += chunk.count(b"\n")
    return count


def iter_jsonl(path: Path):
    """Stream docs from JSONL file."""
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------
if __name__ == "__main__":
    if not CORPUS_PATH.exists():
        raise FileNotFoundError(f"Corpus not found: {CORPUS_PATH}")

    print(f"[INFO] Using device: {DEVICE}")
    print(f"[INFO] Batch size: {BATCH_SIZE}")
    print(f"[INFO] FAISS threads: {NUM_FAISS_THREADS}")
    print(f"[INFO] RAM available: 60GB (IndexIVFPQ required)")

    # Set FAISS to use all CPU cores
    faiss.omp_set_num_threads(NUM_FAISS_THREADS)

    print(f"[INFO] Loading embedding model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME, device=DEVICE)
    model.eval()
    print("[INFO] Model loaded.")

    print(f"[INFO] Counting documents in {CORPUS_PATH} ...")
    total_docs = count_lines_fast(CORPUS_PATH)
    print(f"[INFO] Total documents: {total_docs:,}")

    # Check for checkpoint
    start_from = 0
    if CHECKPOINT_PATH.exists():
        print(f"[INFO] Found checkpoint at {CHECKPOINT_PATH}")
        ckpt = np.load(CHECKPOINT_PATH, allow_pickle=True)
        start_from = int(ckpt['n_seen'])
        print(f"[INFO] Resuming from vector {start_from:,}")
        
        # Load partial index if exists
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
    doc_counter = 0

    print("\n[INFO] Building compressed FAISS index (IVF-PQ) ...")
    print(f"[INFO] Index params: nlist={NLIST}, m={M}, nbits={NBITS}")
    print(f"[INFO] Estimated index size: ~{(total_docs * M * NBITS / 8) / (1024**3):.2f} GB")
    print(f"[INFO] Will train on first {TRAIN_SIZE:,} vectors\n")

    for doc in tqdm(iter_jsonl(CORPUS_PATH), total=total_docs, unit="docs"):
        # Skip already processed docs
        if doc_counter < start_from:
            doc_counter += 1
            continue

        text = (doc.get("text") or "").strip()
        if not text:
            continue

        # meta is written immediately (no need to keep in RAM)
        meta = {
            "doc_id": doc.get("doc_id"),
            "source": doc.get("source"),
            "title": doc.get("title"),
        }
        meta_fout.write(json.dumps(meta, ensure_ascii=False) + "\n")

        batch_texts.append(text)

        if len(batch_texts) >= BATCH_SIZE:
            with torch.inference_mode():
                emb = model.encode(
                    batch_texts,
                    batch_size=BATCH_SIZE,
                    device=DEVICE,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    normalize_embeddings=True,  # cosine via dot product
                )

            if dim is None:
                dim = emb.shape[1]
                print(f"\n[INFO] Embedding dimension: {dim}")
                
                # Create compressed IVF-PQ index
                quantizer = faiss.IndexFlatIP(dim)
                index = faiss.IndexIVFPQ(quantizer, dim, NLIST, M, NBITS)
                print(f"[INFO] Created IndexIVFPQ (memory-efficient)")

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
                    print(f"[INFO] Added {n_seen:,} vectors. Continuing with remaining data...\n")
            else:
                # Phase 2: Add vectors normally after training
                index.add(emb.astype("float32"))
                n_seen += emb.shape[0]

            batch_texts.clear()
            doc_counter += BATCH_SIZE

            # Checkpoint every N vectors
            if n_seen > 0 and n_seen % CHECKPOINT_INTERVAL == 0:
                print(f"\n[INFO] Checkpointing at {n_seen:,} vectors...")
                faiss.write_index(index, str(INDEX_OUT_PATH))
                np.savez(CHECKPOINT_PATH, n_seen=n_seen, is_trained=is_trained)
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
            index.add(emb.astype('float32'))
            n_seen += emb.shape[0]
        
        batch_texts.clear()

    meta_fout.close()

    print(f"\n[INFO] Total indexed vectors: {n_seen:,}")
    print(f"[INFO] Index memory usage: ~{(n_seen * M * NBITS / 8) / (1024**3):.2f} GB")
    print(f"[INFO] Saving FAISS index to: {INDEX_OUT_PATH}")
    faiss.write_index(index, str(INDEX_OUT_PATH))
    
    # Clean up checkpoint
    if CHECKPOINT_PATH.exists():
        CHECKPOINT_PATH.unlink()
        print("[INFO] Checkpoint removed.")
    
    print("[INFO] Done.")