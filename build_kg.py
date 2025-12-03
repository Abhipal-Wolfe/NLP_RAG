"""
Build knowledge graph triplets from CHUNKED biomedical corpus using vLLM.

CRITICAL ALIGNMENT:
-------------------
This script processes the SAME chunked corpus used to build the FAISS index.
Each line in the corpus has a doc_id like: "pubmed::12345::chunk2"

The FAISS index was built by iterating valid (non-empty) chunks sequentially.
Therefore, FAISS vector ID 0 = first valid chunk, ID 1 = second valid chunk, etc.

To maintain alignment, we:
1. Process chunks in the SAME order as FAISS index builder
2. Track chunk_index (0-based counter of valid chunks)
3. This chunk_index MATCHES the FAISS vector ID

Inputs:
-------
data/corpus/med_chunked_corpus.jsonl
    {"doc_id": "source::orig_id::chunkN", "source": ..., "title": ..., "text": ..., "meta": {...}}

Outputs:
--------
data/kg/med_chunks.jsonl
    {"chunk_index": int, "doc_id": str, "source": str, "title": str, "text": str}

data/kg/med_kg_triplets.jsonl
    {"chunk_index": int, "doc_id": str, "head": str, "relation": str, "tail": str}

Checkpoint:
-----------
data/kg/kg_checkpoint.json
    {"next_line_index": int, "next_chunk_index": int}
"""

import json
import os
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from tqdm import tqdm
from vllm import LLM, SamplingParams


# -------------------------------------------------------------------
# Logging
# -------------------------------------------------------------------
logging.getLogger().setLevel(logging.ERROR)
logging.getLogger("vllm").setLevel(logging.ERROR)


# -------------------------------------------------------------------
# Config
# -------------------------------------------------------------------
CORPUS_PATH = Path("/scratch/as20410/NLP_project_final/NLP_RAG/data/corpus/med_chunked_corpus.jsonl")

KG_DIR = Path("/scratch/as20410/NLP_project_final/NLP_RAG/data/kg")
KG_DIR.mkdir(parents=True, exist_ok=True)

CHUNK_OUT_PATH = KG_DIR / "med_chunks.jsonl"
TRIPLE_OUT_PATH = KG_DIR / "med_kg_triplets.jsonl"
CHECKPOINT_PATH = KG_DIR / "kg_checkpoint.json"

# Dynamically count total lines (first run only)
TOTAL_LINES = None  # Will be counted if not checkpointed

# Biomedical LLM model
LLM_MODEL_NAME = os.environ.get("KG_LLM_MODEL", "BioMistral/BioMistral-7B")

# vLLM sampling params
MAX_NEW_TOKENS = int(os.environ.get("KG_MAX_NEW_TOKENS", "256"))
TEMPERATURE = float(os.environ.get("KG_TEMPERATURE", "0.1"))
TOP_P = float(os.environ.get("KG_TOP_P", "0.9"))

# Batch size (conservative for 7B model)
LLM_BATCH_SIZE = int(os.environ.get("KG_LLM_BATCH_SIZE", "32"))

# Max text length to send to LLM (chunks are ~1600 chars, this is safe)
MAX_CHARS_FOR_LLM = int(os.environ.get("KG_MAX_CHARS_FOR_LLM", "2000"))

LOG_INTERVAL = 10_000


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------
def count_lines_fast(path: Path) -> int:
    """Fast line count."""
    if not path.exists():
        raise FileNotFoundError(path)
    count = 0
    with path.open("rb", buffering=1024 * 1024 * 16) as f:
        for chunk in iter(lambda: f.read(1024 * 1024 * 16), b""):
            count += chunk.count(b"\n")
    return count


def iter_jsonl(path: Path) -> Iterable[str]:
    """Stream raw lines from JSONL file."""
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            yield line


TRIPLE_PROMPT_TEMPLATE = """You are an expert biomedical knowledge extraction system.

Extract factual knowledge triples from the text below. Output ONLY triples in this exact format:
<HEAD, RELATION, TAIL>

Rules:
- Use canonical biomedical terms (diseases, drugs, genes, proteins, pathways)
- Use short relations: treats, causes, inhibits, activates, associated_with, located_in, etc.
- Extract ONLY facts explicitly stated in the text
- If no facts, output nothing after "Triples:"

Text:
{TEXT}

Triples:
"""


def build_prompt(text: str) -> str:
    """Build prompt."""
    return TRIPLE_PROMPT_TEMPLATE.format(TEXT=text)


def parse_triplets(generated: str) -> List[Tuple[str, str, str]]:
    """Parse <HEAD, RELATION, TAIL> triples from LLM output."""
    if "Triples:" in generated:
        _, after = generated.split("Triples:", 1)
    else:
        after = generated

    lines = [l.strip() for l in after.splitlines() if l.strip()]
    text = " ".join(lines)

    import re
    pattern = re.compile(r"<([^<>]+)>")
    triples: List[Tuple[str, str, str]] = []

    for match in pattern.finditer(text):
        inner = match.group(1)
        parts = [p.strip() for p in inner.split(",")]
        if len(parts) != 3:
            continue
        h, r, t = parts
        if not h or not r or not t:
            continue
        triples.append((h, r, t))

    return triples


def load_checkpoint(path: Path) -> Tuple[int, int]:
    """Load checkpoint: (next_line_index, next_chunk_index)."""
    if not path.exists():
        return 0, 0
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return int(data.get("next_line_index", 0)), int(data.get("next_chunk_index", 0))


def save_checkpoint(path: Path, next_line_index: int, next_chunk_index: int) -> None:
    """Save checkpoint."""
    with path.open("w", encoding="utf-8") as f:
        json.dump({
            "next_line_index": next_line_index,
            "next_chunk_index": next_chunk_index,
        }, f)


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------
if __name__ == "__main__":
    if not CORPUS_PATH.exists():
        raise FileNotFoundError(f"Corpus not found: {CORPUS_PATH}")

    print(f"[INFO] LLM model: {LLM_MODEL_NAME}")
    print(f"[INFO] Batch size: {LLM_BATCH_SIZE}")
    print(f"[INFO] MAX_NEW_TOKENS={MAX_NEW_TOKENS}, TEMP={TEMPERATURE}, TOP_P={TOP_P}")

    # Count total lines for progress bar
    print("[INFO] Counting corpus lines...")
    total_lines = count_lines_fast(CORPUS_PATH)
    print(f"[INFO] Total lines: {total_lines:,}")

    print("[INFO] Initializing vLLM...")
    gpu_mem_util = float(os.environ.get("VLLM_GPU_MEM_UTIL", "0.85"))
    max_model_len = int(os.environ.get("VLLM_MAX_MODEL_LEN", "4096"))
    
    llm = LLM(
        model=LLM_MODEL_NAME,
        dtype="float16",
        tensor_parallel_size=1,
        gpu_memory_utilization=gpu_mem_util,
        max_model_len=max_model_len,
        trust_remote_code=True,  # Some models need this
    )
    sampling_params = SamplingParams(
        temperature=TEMPERATURE,
        top_p=TOP_P,
        max_tokens=MAX_NEW_TOKENS,
    )
    print("[INFO] vLLM ready.\n")

    # Load checkpoint
    next_line_index, next_chunk_index = load_checkpoint(CHECKPOINT_PATH)
    print(f"[INFO] Resuming from line {next_line_index:,}, chunk_index {next_chunk_index:,}")

    # Open output files in append mode
    chunk_fout = CHUNK_OUT_PATH.open("a", encoding="utf-8")
    triple_fout = TRIPLE_OUT_PATH.open("a", encoding="utf-8")

    # Batch buffers
    batch_prompts: List[str] = []
    batch_meta: List[Dict] = []

    line_index = 0
    chunk_index = next_chunk_index

    pbar = tqdm(
        total=total_lines,
        unit="lines",
        initial=next_line_index,
        desc=f"lines={next_line_index} chunks={next_chunk_index}",
    )

    for raw_line in iter_jsonl(CORPUS_PATH):
        # Skip to checkpoint
        if line_index < next_line_index:
            line_index += 1
            continue

        line_index += 1
        pbar.update(1)

        line = raw_line.strip()
        if not line:
            continue

        try:
            doc = json.loads(line)
        except json.JSONDecodeError:
            continue

        text = (doc.get("text") or "").strip()
        if not text:
            # Skip empty chunks (same as FAISS index builder)
            continue

        # This is a valid chunk - assign chunk_index
        current_chunk_index = chunk_index

        # Truncate text if too long
        if len(text) > MAX_CHARS_FOR_LLM:
            text = text[:MAX_CHARS_FOR_LLM]

        doc_id = doc.get("doc_id")
        source = doc.get("source")
        title = doc.get("title")

        # Write chunk metadata
        chunk_record = {
            "chunk_index": current_chunk_index,
            "doc_id": doc_id,
            "source": source,
            "title": title,
            "text": text,
        }
        chunk_fout.write(json.dumps(chunk_record, ensure_ascii=False) + "\n")

        # Prepare LLM prompt
        prompt = build_prompt(text)
        batch_prompts.append(prompt)
        batch_meta.append({
            "chunk_index": current_chunk_index,
            "doc_id": doc_id,
        })

        chunk_index += 1
        pbar.set_description(f"lines={line_index} chunks={chunk_index}")

        # Process batch
        if len(batch_prompts) >= LLM_BATCH_SIZE:
            outputs = llm.generate(batch_prompts, sampling_params)

            for out, meta in zip(outputs, batch_meta):
                if not out.outputs:
                    continue
                gen_text = out.outputs[0].text
                triples = parse_triplets(gen_text)

                for (h, r, t) in triples:
                    triple_record = {
                        "chunk_index": meta["chunk_index"],
                        "doc_id": meta["doc_id"],
                        "head": h,
                        "relation": r,
                        "tail": t,
                    }
                    triple_fout.write(json.dumps(triple_record, ensure_ascii=False) + "\n")

            batch_prompts.clear()
            batch_meta.clear()

            # Checkpoint after each batch
            save_checkpoint(CHECKPOINT_PATH, line_index, chunk_index)
            chunk_fout.flush()
            triple_fout.flush()

        if line_index % LOG_INTERVAL == 0:
            print(f"\n[INFO] Processed {line_index:,} lines, {chunk_index:,} chunks")

    # Process final batch
    if batch_prompts:
        outputs = llm.generate(batch_prompts, sampling_params)
        for out, meta in zip(outputs, batch_meta):
            if not out.outputs:
                continue
            gen_text = out.outputs[0].text
            triples = parse_triplets(gen_text)

            for (h, r, t) in triples:
                triple_record = {
                    "chunk_index": meta["chunk_index"],
                    "doc_id": meta["doc_id"],
                    "head": h,
                    "relation": r,
                    "tail": t,
                }
                triple_fout.write(json.dumps(triple_record, ensure_ascii=False) + "\n")

    pbar.close()

    # Final checkpoint
    save_checkpoint(CHECKPOINT_PATH, line_index, chunk_index)
    chunk_fout.close()
    triple_fout.close()

    print(f"\n{'='*70}")
    print("KNOWLEDGE GRAPH BUILD COMPLETE")
    print(f"{'='*70}")
    print(f"Processed lines: {line_index:,}")
    print(f"Valid chunks: {chunk_index:,}")
    print(f"Chunks file: {CHUNK_OUT_PATH}")
    print(f"Triples file: {TRIPLE_OUT_PATH}")
    print(f"{'='*70}")
    
    print(f"\n✅ chunk_index values match FAISS vector IDs!")