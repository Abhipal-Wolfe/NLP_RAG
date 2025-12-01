# scripts/build_kg_from_corpus_vllm.py
"""
Build a knowledge graph (triples) from a biomedical JSONL corpus using vLLM.

Alignment with FAISS:
----------------------
- The FAISS index script encodes one embedding per JSONL line (doc) with non-empty "text".
- Here we follow the SAME rule:
    * 1 line (doc) with non-empty text = 1 "chunk"
    * chunk_id = 0-based index over docs with non-empty text
- Therefore, `chunk_id` in this script == FAISS vector ID (if built from the same corpus).

Inputs:
-------
../data/corpus/med_pure_corpus.jsonl
    Each line:
    {
        "doc_id": ...,
        "source": ...,
        "title": ...,
        "text": ...
    }

Outputs:
--------
data/kg/med_chunks.jsonl
    {"chunk_id": int, "doc_id": str, "source": str, "title": str, "text": str}

data/kg/med_kg_triplets.jsonl
    {"chunk_id": int, "doc_id": str,
     "head": str, "relation": str, "tail": str}

Checkpoint:
-----------
data/kg/kg_checkpoint.json
    {
        "next_line_index": int,   # next JSONL line index to process (0-based)
        "next_chunk_id": int      # next chunk_id to assign
    }
"""

import json
import os
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from tqdm import tqdm
from vllm import LLM, SamplingParams


# -------------------------------------------------------------------
# Logging: suppress vLLM and other noisy logs
# -------------------------------------------------------------------
logging.getLogger().setLevel(logging.ERROR)
logging.getLogger("vllm").setLevel(logging.ERROR)
logging.getLogger("vllm.engine").setLevel(logging.ERROR)
logging.getLogger("vllm.worker").setLevel(logging.ERROR)


# -------------------------------------------------------------------
# Config
# -------------------------------------------------------------------
CORPUS_PATH = Path("../data/corpus/med_pure_corpus.jsonl")

KG_DIR = Path("data/kg")
KG_DIR.mkdir(parents=True, exist_ok=True)

CHUNK_OUT_PATH = KG_DIR / "med_chunks.jsonl"
TRIPLE_OUT_PATH = KG_DIR / "med_kg_triplets.jsonl"
CHECKPOINT_PATH = KG_DIR / "kg_checkpoint.json"

# Total number of lines is fixed (user-provided)
TOTAL_LINES = 24_062_518

# Strong, free biomedical / scientific LLM (can be changed via env on HPC)
# Example: "BioMistral/BioMistral-7B" (adjust as needed)
LLM_MODEL_NAME = os.environ.get("KG_LLM_MODEL", "BioMistral/BioMistral-7B")

# vLLM sampling params
MAX_NEW_TOKENS = int(os.environ.get("KG_MAX_NEW_TOKENS", "256"))
TEMPERATURE = float(os.environ.get("KG_TEMPERATURE", "0.1"))
TOP_P = float(os.environ.get("KG_TOP_P", "0.9"))

# vLLM batch size (default larger for stronger HPC GPUs; tune via env)
LLM_BATCH_SIZE = int(os.environ.get("KG_LLM_BATCH_SIZE", "64"))

# Interval (in docs) for printing textual progress logs (tqdm shows progress bar)
LOG_INTERVAL_DOCS = 10_000


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------
def iter_jsonl(path: Path) -> Iterable[str]:
    """Stream raw lines from JSONL file (for index-based resume)."""
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            yield line


TRIPLE_PROMPT_TEMPLATE = """You are an information extraction system for biomedical scientific text.

Extract concise factual knowledge triples from the text below.
Each triple MUST follow this exact format (no extra text):

<HEAD, RELATION, TAIL>,
<HEAD, RELATION, TAIL>,
...

Rules:
- Use short, canonical biomedical entity names (diseases, drugs, genes, pathways, etc.).
- Use informative but short relations (e.g., "treats", "causes", "associated_with", "inhibits", "encodes", "expressed_in").
- Do NOT invent facts that are not supported by the text.
- If no useful facts, output an empty line after "Triplets:".

Text:
{TEXT}

Triplets:
"""


def build_prompt(text: str) -> str:
    """Build prompt by inserting the text into the template."""
    return TRIPLE_PROMPT_TEMPLATE.format(TEXT=text)


def parse_triplets(generated: str) -> List[Tuple[str, str, str]]:
    """
    Parse <HEAD, RELATION, TAIL> triples from LLM output.
    Robust to minor formatting noise.
    """
    # Many models echo the prompt; focus on the part after 'Triplets:'
    if "Triplets:" in generated:
        _, after = generated.split("Triplets:", 1)
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
    """
    Load checkpoint.
    Returns:
        next_line_index (int): JSONL line index to process next (0-based)
        next_chunk_id  (int): chunk_id to assign next
    """
    if not path.exists():
        return 0, 0
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return int(data.get("next_line_index", 0)), int(data.get("next_chunk_id", 0))


def save_checkpoint(path: Path, next_line_index: int, next_chunk_id: int) -> None:
    """Save checkpoint to disk."""
    tmp = {
        "next_line_index": next_line_index,
        "next_chunk_id": next_chunk_id,
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(tmp, f)


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------
if __name__ == "__main__":
    if not CORPUS_PATH.exists():
        raise FileNotFoundError(f"Corpus not found: {CORPUS_PATH}")

    print(f"[INFO] LLM model: {LLM_MODEL_NAME}")
    print(f"[INFO] vLLM batch size: {LLM_BATCH_SIZE}")
    print(f"[INFO] MAX_NEW_TOKENS={MAX_NEW_TOKENS}, TEMP={TEMPERATURE}, TOP_P={TOP_P}")

    print("[INFO] Initializing vLLM LLM ...")

    # Default settings assume a reasonably strong HPC GPU node.
    # You can override these via environment variables.
    gpu_mem_util = float(os.environ.get("VLLM_GPU_MEM_UTIL", "0.9"))
    max_model_len = int(os.environ.get("VLLM_MAX_MODEL_LEN", "4096"))
    swap_space_gb = int(os.environ.get("VLLM_SWAP_SPACE_GB", "4"))
    tp_size = int(os.environ.get("VLLM_TP_SIZE", "1"))  # tensor parallel across multiple GPUs if desired

    llm = LLM(
        model=LLM_MODEL_NAME,
        dtype="float16",                # FP16 on modern HPC GPUs
        tensor_parallel_size=tp_size,   # use >1 if you want to span multiple GPUs
        gpu_memory_utilization=gpu_mem_util,
        max_model_len=max_model_len,
        swap_space=swap_space_gb,
    )
    sampling_params = SamplingParams(
        temperature=TEMPERATURE,
        top_p=TOP_P,
        max_tokens=MAX_NEW_TOKENS,
    )
    print("[INFO] vLLM LLM ready.")

    # Use fixed total line count for progress bar
    total_lines = TOTAL_LINES
    print(f"[INFO] Total lines (docs) in corpus (fixed): {total_lines:,}")

    # Load checkpoint (line index, chunk_id)
    next_line_index, next_chunk_id = load_checkpoint(CHECKPOINT_PATH)
    print(f"[INFO] Resuming from line_index >= {next_line_index}, chunk_id >= {next_chunk_id}")

    # Output files in append mode (can resume)
    chunk_fout = CHUNK_OUT_PATH.open("a", encoding="utf-8")
    triple_fout = TRIPLE_OUT_PATH.open("a", encoding="utf-8")

    # Batch buffers
    batch_prompts: List[str] = []
    batch_meta: List[Dict] = []  # holds {chunk_id, doc_id, ...} per prompt

    # Iterate over corpus lines with an explicit line index
    line_iter = iter_jsonl(CORPUS_PATH)
    line_index = 0    # 0-based line counter
    chunk_id = next_chunk_id  # current chunk_id to assign for non-empty texts

    # Global progress bar (the only tqdm instance)
    pbar = tqdm(
        total=total_lines,
        unit="lines",
        initial=next_line_index,
        desc=f"lines={next_line_index} chunks={chunk_id}",
    )

    for raw_line in line_iter:
        # Skip lines until we reach checkpoint line_index
        if line_index < next_line_index:
            line_index += 1
            # pbar already starts from initial=next_line_index, so no update here
            continue

        line_index += 1
        pbar.update(1)

        line = raw_line.strip()
        if not line:
            continue

        try:
            doc = json.loads(line)
        except json.JSONDecodeError:
            # Skip malformed line
            continue

        text = (doc.get("text") or "").strip()
        if not text:
            # Same as FAISS script: documents with empty text are skipped entirely
            continue

        # Optional: limit text length sent to the LLM to reduce memory usage
        max_chars_for_llm = int(os.environ.get("KG_MAX_CHARS_FOR_LLM", "4000"))
        if len(text) > max_chars_for_llm:
            text = text[:max_chars_for_llm]

        # FAISS: each non-empty text document -> one embedding.
        # Here: each non-empty text document -> one "chunk".
        current_chunk_id = chunk_id

        doc_id = doc.get("doc_id")
        source = doc.get("source")
        title = doc.get("title")

        # Write chunk metadata immediately
        chunk_record = {
            "chunk_id": current_chunk_id,
            "doc_id": doc_id,
            "source": source,
            "title": title,
            "text": text,
        }
        chunk_fout.write(json.dumps(chunk_record, ensure_ascii=False) + "\n")

        # Prepare LLM prompt
        prompt = build_prompt(text)
        batch_prompts.append(prompt)
        batch_meta.append(
            {
                "chunk_id": current_chunk_id,
                "doc_id": doc_id,
            }
        )

        chunk_id += 1  # prepare id for next non-empty document

        # Update progress bar description to show current position
        pbar.set_description(f"lines={line_index} chunks={chunk_id}")

        # If batch is full, run vLLM
        if len(batch_prompts) >= LLM_BATCH_SIZE:
            outputs = llm.generate(batch_prompts, sampling_params)

            # vLLM guarantees: outputs are in the same order as prompts
            for out, meta in zip(outputs, batch_meta):
                if not out.outputs:
                    continue
                gen_text = out.outputs[0].text
                triples = parse_triplets(gen_text)

                for (h, r, t) in triples:
                    triple_record = {
                        "chunk_id": meta["chunk_id"],
                        "doc_id": meta["doc_id"],
                        "head": h,
                        "relation": r,
                        "tail": t,
                    }
                    triple_fout.write(json.dumps(triple_record, ensure_ascii=False) + "\n")

            # flush & clear batch
            batch_prompts.clear()
            batch_meta.clear()

            # checkpoint after each batch
            save_checkpoint(CHECKPOINT_PATH, line_index, chunk_id)
            chunk_fout.flush()
            triple_fout.flush()

        # Optional: coarse-grained textual progress log (you can remove this if not needed)
        if line_index % LOG_INTERVAL_DOCS == 0:
            print(f"[INFO] Processed {line_index:,} lines, current chunk_id={chunk_id:,}")

    # Process final partial batch
    if batch_prompts:
        outputs = llm.generate(batch_prompts, sampling_params)
        for out, meta in zip(outputs, batch_meta):
            if not out.outputs:
                continue
            gen_text = out.outputs[0].text
            triples = parse_triplets(gen_text)

            for (h, r, t) in triples:
                triple_record = {
                    "chunk_id": meta["chunk_id"],
                    "doc_id": meta["doc_id"],
                    "head": h,
                    "relation": r,
                    "tail": t,
                }
                triple_fout.write(json.dumps(triple_record, ensure_ascii=False) + "\n")

        batch_prompts.clear()
        batch_meta.clear()

    pbar.close()

    # Final checkpoint & close
    save_checkpoint(CHECKPOINT_PATH, line_index, chunk_id)
    chunk_fout.close()
    triple_fout.close()

    print(f"\n[INFO] Finished KG build.")
    print(f"[INFO] Last processed line_index = {line_index}")
    print(f"[INFO] Last assigned chunk_id    = {chunk_id - 1}")
    print(f"[INFO] Chunks written to         : {CHUNK_OUT_PATH}")
    print(f"[INFO] Triples written to        : {TRIPLE_OUT_PATH}")
    print(f"[INFO] Checkpoint saved at       : {CHECKPOINT_PATH}")