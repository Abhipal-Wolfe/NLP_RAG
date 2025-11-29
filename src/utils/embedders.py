"""Query embedding utilities"""

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm


def query_encode(input_list):
    """Encode queries using MedCPT-Query-Encoder"""
    model = AutoModel.from_pretrained("ncbi/MedCPT-Query-Encoder")
    if torch.cuda.is_available():
        model = model.to(0)
    tokenizer = AutoTokenizer.from_pretrained("ncbi/MedCPT-Query-Encoder")

    queries = []
    splits = [i for i in range(0, len(input_list), 100)]

    for i in tqdm(splits, desc="Encoding queries", disable=True):
        split_queries = input_list[i:i+100]
        with torch.no_grad():
            encoded = tokenizer(
                split_queries,
                truncation=True,
                padding=True,
                return_tensors='pt',
                max_length=192,
            )
            encoded = {key: tensor.to(0) for key, tensor in encoded.items()}
            embeds = model(**encoded).last_hidden_state[:, 0, :]
            query_embeddings = embeds.detach().cpu().numpy()
            queries.extend(query_embeddings)

    return np.vstack(queries)
