import json
from pathlib import Path
from typing import Dict, Any, List
from concurrent.futures import ProcessPoolExecutor, as_completed

from datasets import load_dataset
from tqdm import tqdm

# -------------------------------------------------------------------
# Config
# -------------------------------------------------------------------
CORPUS_DIR = Path("data/corpus")
CORPUS_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = CORPUS_DIR / "med_chunked_corpus.jsonl"

# Chunking parameters
CHUNK_SIZE = 400  # tokens per chunk (safe for 512 token models)
CHUNK_OVERLAP = 50  # token overlap between chunks
CHARS_PER_TOKEN = 4  # rough approximation for English medical text

# Limit per source (None = use all rows)
MAX_DOCS_PER_SOURCE = {
    "pubmed": None,
    "textbook": None,
    "guideline": None,
}

# Number of parallel shards for PubMed
PUBMED_NUM_SHARDS = 8


# -------------------------------------------------------------------
# Chunking Helper
# -------------------------------------------------------------------
def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """
    Split text into overlapping chunks based on character approximation.
    
    Args:
        text: Input text to chunk
        chunk_size: Target tokens per chunk
        overlap: Overlapping tokens between chunks
    
    Returns:
        List of text chunks
    """
    # Convert token counts to character counts
    chunk_chars = chunk_size * CHARS_PER_TOKEN
    overlap_chars = overlap * CHARS_PER_TOKEN
    
    if len(text) <= chunk_chars:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_chars
        
        # If this is not the last chunk, try to break at sentence boundary
        if end < len(text):
            # Look for sentence ending in the last 20% of chunk
            search_start = end - int(chunk_chars * 0.2)
            sentence_ends = ['. ', '.\n', '? ', '?\n', '! ', '!\n']
            
            best_break = -1
            for sent_end in sentence_ends:
                idx = text.rfind(sent_end, search_start, end)
                if idx > best_break:
                    best_break = idx + len(sent_end)
            
            if best_break > search_start:
                end = best_break
        
        chunk_piece = text[start:end].strip()
        if chunk_piece:
            chunks.append(chunk_piece)
        
        # Move start forward, accounting for overlap
        start = end - overlap_chars
        
        # Prevent infinite loop on very short texts
        if start <= 0 and chunks:
            break
    
    return chunks


# -------------------------------------------------------------------
# PubMed (parallel, sharded, WITH CHUNKING)
# -------------------------------------------------------------------
def process_pubmed_shard(
    shard_idx: int,
    num_shards: int,
    max_docs: int | None,
) -> str:
    """
    Process a shard of MedRAG/pubmed with chunking.
    Each HF row -> MULTIPLE JSONL documents (one per chunk).
    """
    hf_id = "MedRAG/pubmed"
    split = "train"
    source_name = "pubmed"

    out_path = CORPUS_DIR / f"med_chunked_pubmed_shard{shard_idx}.jsonl"
    print(f"\n=== [pubmed-shard-{shard_idx}] Processing with chunking ===")

    ds_stream = load_dataset(hf_id, split=split, streaming=True)
    ds = ds_stream.shard(num_shards=num_shards, index=shard_idx)

    iter_rows = 0
    written_chunks = 0

    with out_path.open("w", encoding="utf-8") as fout:
        for ex in tqdm(ds, desc=f"pubmed shard {shard_idx}", unit="rows"):
            if isinstance(max_docs, int) and iter_rows >= max_docs:
                break
            iter_rows += 1

            raw_text = (ex.get("contents") or "").strip()
            if not raw_text:
                continue

            title = (ex.get("title") or "").strip()
            pmid = ex.get("PMID") or f"shard{shard_idx}_row{iter_rows}"
            orig_id = str(pmid)

            # Chunk the text
            chunks = chunk_text(raw_text)
            
            for chunk_idx, chunk_content in enumerate(chunks):
                meta: Dict[str, Any] = {
                    "original_doc_id": orig_id,
                    "chunk_index": chunk_idx,
                    "total_chunks": len(chunks),
                }
                if "id" in ex:
                    meta["id"] = ex["id"]

                doc = {
                    "doc_id": f"{source_name}::{orig_id}::chunk{chunk_idx}",
                    "source": source_name,
                    "title": title,
                    "text": chunk_content,
                    "meta": meta,
                }
                fout.write(json.dumps(doc, ensure_ascii=False) + "\n")
                written_chunks += 1

    print(f"[pubmed-shard-{shard_idx}] processed {iter_rows} docs -> {written_chunks} chunks")
    return str(out_path)


# -------------------------------------------------------------------
# Other sources (sequential, WITH CHUNKING)
# -------------------------------------------------------------------
def write_docs_for_source(
    hf_id: str,
    split: str,
    source_name: str,
    text_field: str,
    title_field: str | None,
    id_field: str | None,
    max_docs: int | None,
    extra_meta_fields: list[str] | None = None,
):
    """
    Stream a Hugging Face dataset and append chunked documents to OUT_PATH.
    Each HF row -> MULTIPLE JSONL documents (one per chunk).
    """
    print(f"\n=== Loading {source_name} from {hf_id} ({split}) with chunking ===")

    ds = load_dataset(hf_id, split=split, streaming=True)

    iter_rows = 0
    written_chunks = 0

    with OUT_PATH.open("a", encoding="utf-8") as fout:
        for ex in tqdm(ds, desc=f"{source_name} rows", unit="rows"):
            if isinstance(max_docs, int) and iter_rows >= max_docs:
                break
            iter_rows += 1

            raw_text = (ex.get(text_field) or "").strip()
            if not raw_text:
                continue

            title = (ex.get(title_field) or "").strip() if title_field else ""
            orig_id = str(ex.get(id_field) or iter_rows)

            # Chunk the text
            chunks = chunk_text(raw_text)
            
            for chunk_idx, chunk_content in enumerate(chunks):
                meta: Dict[str, Any] = {
                    "original_doc_id": orig_id,
                    "chunk_index": chunk_idx,
                    "total_chunks": len(chunks),
                }
                
                # Add extra metadata
                if extra_meta_fields:
                    for k in extra_meta_fields:
                        if k in ex:
                            meta[k] = ex[k]

                doc = {
                    "doc_id": f"{source_name}::{orig_id}::chunk{chunk_idx}",
                    "source": source_name,
                    "title": title,
                    "text": chunk_content,
                    "meta": meta,
                }
                fout.write(json.dumps(doc, ensure_ascii=False) + "\n")
                written_chunks += 1

    print(f"[{source_name}] processed {iter_rows} docs -> {written_chunks} chunks")


# -------------------------------------------------------------------
# Merge helpers
# -------------------------------------------------------------------
def merge_files(part_files: list[str], final_path: Path) -> None:
    """Concatenate JSONL part files into one file."""
    with final_path.open("a", encoding="utf-8") as fout:
        for p in part_files:
            src = Path(p)
            with src.open("r", encoding="utf-8") as fin:
                for line in fin:
                    fout.write(line)
    print(f"[MERGE] Merged {len(part_files)} part files into {final_path}")


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------
if __name__ == "__main__":
    # Start with a fresh unified corpus file
    if OUT_PATH.exists():
        print(f"Removing existing corpus file: {OUT_PATH}")
        OUT_PATH.unlink()

    print(f"\n{'='*70}")
    print(f"CHUNKING CONFIG:")
    print(f"  Chunk size: {CHUNK_SIZE} tokens (~{CHUNK_SIZE * CHARS_PER_TOKEN} chars)")
    print(f"  Overlap: {CHUNK_OVERLAP} tokens (~{CHUNK_OVERLAP * CHARS_PER_TOKEN} chars)")
    print(f"  This will create MORE vectors but enable retrieval from ENTIRE documents")
    print(f"{'='*70}\n")

    # ---------------- PubMed in parallel (WITH CHUNKING) ----------------
    print("\n=== PubMed (MedRAG/pubmed) - parallel shards with chunking ===")
    pubmed_part_files: list[str] = []

    with ProcessPoolExecutor(max_workers=PUBMED_NUM_SHARDS) as executor:
        futures = []
        for shard_idx in range(PUBMED_NUM_SHARDS):
            futures.append(
                executor.submit(
                    process_pubmed_shard,
                    shard_idx=shard_idx,
                    num_shards=PUBMED_NUM_SHARDS,
                    max_docs=MAX_DOCS_PER_SOURCE["pubmed"],
                )
            )

        for fut in as_completed(futures):
            try:
                part_path = fut.result()
                pubmed_part_files.append(part_path)
            except Exception as e:
                print(f"[ERROR] PubMed shard failed: {e}")

    # Merge PubMed shards into OUT_PATH
    if pubmed_part_files:
        merge_files(pubmed_part_files, OUT_PATH)
    else:
        print("[WARN] No PubMed shard files produced.")

    # ---------------- Textbooks (WITH CHUNKING) ----------------
    write_docs_for_source(
        hf_id="MedRAG/textbooks",
        split="train",
        source_name="textbook",
        text_field="contents",
        title_field="title",
        id_field="id",
        max_docs=MAX_DOCS_PER_SOURCE["textbook"],
        extra_meta_fields=None,
    )

    # ---------------- Clinical Guidelines (WITH CHUNKING) ----------------
    write_docs_for_source(
        hf_id="epfl-llm/guidelines",
        split="train",
        source_name="guideline",
        text_field="clean_text",
        title_field="title",
        id_field="id",
        max_docs=MAX_DOCS_PER_SOURCE["guideline"],
        extra_meta_fields=["source", "url"],
    )

    print(f"\n✅ Done. Chunked corpus written to: {OUT_PATH}")
    
    # Print statistics
    print("\n" + "="*70)
    print("CHUNKING SUMMARY:")
    print("="*70)
    
    chunk_count = 0
    with OUT_PATH.open("r") as f:
        for _ in f:
            chunk_count += 1
    
    print(f"Total chunks created: {chunk_count:,}")
    print(f"Estimated index size: ~{(chunk_count * 768 * 4) / (1024**3):.2f} GB (FlatIP)")
    print(f"With IVF-PQ compression: ~{(chunk_count * 64 * 8 / 8) / (1024**3):.2f} GB")