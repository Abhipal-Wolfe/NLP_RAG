import os
import json
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import logging

# -----------------------------
# Chunking logic: 128 words w/ 32 overlap
# -----------------------------
def chunk_text(text, max_len=128, overlap_len=32):
    words = text.split()
    chunks = []
    if len(words) <= max_len:
        chunks.append(text)
    else:
        for i in range(0, len(words), max_len - overlap_len):
            chunks.append(" ".join(words[i:i+max_len]))
            if i + max_len >= len(words):
                break
    return chunks

# -----------------------------
# Encode chunks
# -----------------------------
def encode_chunks(chunks, model, model_type="medcpt", batch_size=100):
    """
    Encode chunks with the specified model.
    model_type: "medcpt" or "sentencebert"
    """
    if model_type.lower() == "sentencebert":
        # Sentence-BERT handles batching internally
        embeddings = model.encode(
            chunks,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        return embeddings

    elif model_type.lower() == "medcpt":
        import transformers
        embeddings_list = []
        tokenizer = model["tokenizer"]
        net = model["model"]
        device = model["device"]

        for i in tqdm(range(0, len(chunks), batch_size), desc="Encoding batches"):
            batch = chunks[i:i+batch_size]
            with torch.no_grad():
                encoded = tokenizer(batch, truncation=True, padding=True, return_tensors='pt', max_length=512)
                encoded = {k: v.to(device) for k, v in encoded.items()}
                output = net(**encoded).last_hidden_state[:, 0, :]  # CLS token
                embeddings_list.append(output.detach().cpu().numpy())
        return np.vstack(embeddings_list)

    else:
        raise ValueError(f"Unknown model_type {model_type}")

# -----------------------------
# Process all files and merge
# -----------------------------
def process_and_merge(input_dir, output_dir, model, model_type="medcpt", batch_size=100):
    os.makedirs(output_dir, exist_ok=True)
    all_embeddings = []
    total_mapping = {}
    chunk_counter = 0

    files = [f for f in os.listdir(input_dir) if f.endswith(".txt")]
    for file_name in files:
        file_path = os.path.join(input_dir, file_name)
        article_id = file_name.replace(".txt", "")
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()

        chunks = chunk_text(text)
        logging.info(f"{file_name} -> {len(chunks)} chunks")

        embeddings = encode_chunks(chunks, model, model_type=model_type, batch_size=batch_size)
        all_embeddings.append(embeddings)

        # Update mapping: each chunk index → article_id
        for _ in range(len(chunks)):
            total_mapping[chunk_counter] = article_id
            chunk_counter += 1

    # Save merged embeddings and mapping
    total_embeddings = np.vstack(all_embeddings)
    np.save(os.path.join(output_dir, "total.npy"), total_embeddings)
    with open(os.path.join(output_dir, "total.json"), "w", encoding="utf-8") as f:
        json.dump(total_mapping, f, indent=2)

    logging.info(f"Saved merged embeddings to total.npy ({total_embeddings.shape})")
    logging.info(f"Saved mapping to total.json ({len(total_mapping)} entries)")

# -----------------------------
# Script entry point
# -----------------------------
if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True, help="Directory with raw PMC text files (.txt)")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save merged embeddings")
    parser.add_argument("--model_type", type=str, default="medcpt", choices=["medcpt", "sentencebert"], help="Which encoder to use")
    parser.add_argument("--gpu", type=int, default=0, help="GPU index to use if available")
    parser.add_argument("--batch_size", type=int, default=100, help="Batch size for encoding")
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    # Load model
    if args.model_type.lower() == "sentencebert":
        model = SentenceTransformer("sentence-transformers/all-distilroberta-v1", device=device)
    elif args.model_type.lower() == "medcpt":
        from transformers import AutoTokenizer, AutoModel
        tokenizer = AutoTokenizer.from_pretrained("ncbi/MedCPT-Article-Encoder")
        net = AutoModel.from_pretrained("ncbi/MedCPT-Article-Encoder").to(device)
        model = {"model": net, "tokenizer": tokenizer, "device": device}
    else:
        raise ValueError("Invalid model_type")

    process_and_merge(args.input_dir, args.output_dir, model, model_type=args.model_type, batch_size=args.batch_size)