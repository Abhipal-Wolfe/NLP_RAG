# scripts/prepare_med_corpus.py
import json
from pathlib import Path
from typing import Dict, Any
from concurrent.futures import ProcessPoolExecutor, as_completed

from datasets import load_dataset
from tqdm import tqdm

# -------------------------------------------------------------------
# Config
# -------------------------------------------------------------------
CORPUS_DIR = Path("data/corpus")
CORPUS_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = CORPUS_DIR / "med_pure_corpus.jsonl"

# Limit per source (None = use all rows)
MAX_DOCS_PER_SOURCE = {
    "pubmed": None,      # use all rows
    "textbook": None,
    "guideline": None,
}

# Number of parallel shards for PubMed
PUBMED_NUM_SHARDS = 8  # adjust to 2/4/8 depending on your CPU cores


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------
def get_dataset_size(hf_id: str, split: str) -> int | None:
    """
    Try to get number of rows for a dataset split (non-streaming).
    Returns None if it fails, but script can still continue.
    """
    try:
        ds = load_dataset(hf_id, split=split)
        n = len(ds)
        print(f"[INFO] {hf_id} ({split}) has {n} rows.")
        return n
    except Exception as e:
        print(f"[WARN] Could not get dataset size for {hf_id}:{split} ({e}). Proceeding without total.")
        return None


# -------------------------------------------------------------------
# PubMed (parallel, sharded, NO CHUNKING)
# -------------------------------------------------------------------
def process_pubmed_shard(
    shard_idx: int,
    num_shards: int,
    max_docs: int | None,
) -> str:
    """
    Process a shard of MedRAG/pubmed in a separate process.
    Each HF row -> ONE JSONL document (no chunking).
    Writes to data/corpus/med_pure_pubmed_shard{shard_idx}.jsonl
    Returns the output file path (as string).
    """
    hf_id = "MedRAG/pubmed"
    split = "train"
    source_name = "pubmed"

    out_path = CORPUS_DIR / f"med_pure_pubmed_shard{shard_idx}.jsonl"
    print(f"\n=== [pubmed-shard-{shard_idx}] Loading from {hf_id} ({split}), shard {shard_idx}/{num_shards} ===")

    # Streaming dataset, then shard it so each process gets disjoint rows
    ds_stream = load_dataset(hf_id, split=split, streaming=True)
    ds = ds_stream.shard(num_shards=num_shards, index=shard_idx)

    iter_rows = 0
    written_docs = 0

    # We don't know exact size per shard; tqdm will just count rows
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

            meta: Dict[str, Any] = {}
            if "id" in ex:
                meta["id"] = ex["id"]

            doc = {
                "doc_id": f"{source_name}::{orig_id}",
                "source": source_name,
                "title": title,
                "text": raw_text,   # full contents, no chunking
                "meta": meta,
            }
            fout.write(json.dumps(doc, ensure_ascii=False) + "\n")
            written_docs += 1

    print(
        f"[pubmed-shard-{shard_idx}] iterated {iter_rows} rows "
        f"and wrote {written_docs} documents to {out_path}"
    )
    return str(out_path)


# -------------------------------------------------------------------
# Other sources (sequential, NO CHUNKING)
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
    Stream a Hugging Face dataset and append documents to OUT_PATH.
    Each HF row -> ONE JSONL document (no chunking).
    Fields:
      - doc_id: "{source_name}::{orig_id}"
      - source: source_name
      - title: optional
      - text: full text_field
      - meta: dict
    """
    print(f"\n=== Loading {source_name} from {hf_id} ({split}) ===")

    # 1) Get dataset size (non-streaming) for logging and progress bar
    total_rows = get_dataset_size(hf_id, split)

    # 2) Create streaming dataset for actual processing
    ds = load_dataset(hf_id, split=split, streaming=True)

    iter_rows = 0    # how many rows we iterated
    written_docs = 0 # how many documents we wrote

    # 3) Decide tqdm total
    if isinstance(max_docs, int):
        total_for_tqdm = min(max_docs, total_rows) if isinstance(total_rows, int) else max_docs
    else:
        total_for_tqdm = total_rows if isinstance(total_rows, int) else None

    # 4) Iterate and write docs (append to OUT_PATH)
    with OUT_PATH.open("a", encoding="utf-8") as fout:
        for ex in tqdm(
            ds,
            desc=f"{source_name} rows",
            unit="rows",
            total=total_for_tqdm,
        ):
            if isinstance(max_docs, int) and iter_rows >= max_docs:
                break
            iter_rows += 1

            raw_text = (ex.get(text_field) or "").strip()
            if not raw_text:
                continue

            title = (ex.get(title_field) or "").strip() if title_field else ""
            orig_id = str(ex.get(id_field) or iter_rows)

            # optional extra metadata
            meta: Dict[str, Any] = {}
            if extra_meta_fields:
                for k in extra_meta_fields:
                    if k in ex:
                        meta[k] = ex[k]

            doc = {
                "doc_id": f"{source_name}::{orig_id}",
                "source": source_name,
                "title": title,
                "text": raw_text,  # full text
                "meta": meta,
            }
            fout.write(json.dumps(doc, ensure_ascii=False) + "\n")
            written_docs += 1

    print(
        f"[{source_name}] iterated {iter_rows} rows "
        f"and wrote {written_docs} documents to {OUT_PATH}"
    )


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

    # ---------------- PubMed in parallel (no chunk) ----------------
    print("\n=== PubMed (MedRAG/pubmed) will be processed in parallel shards (no chunking) ===")
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

    # ---------------- Textbooks (sequential, no chunk) ----------------
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

    # ---------------- Clinical Guidelines (sequential, no chunk) -----
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

    print(f"\n✅ Done. Unified corpus written to: {OUT_PATH}")
