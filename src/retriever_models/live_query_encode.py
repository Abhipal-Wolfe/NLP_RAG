# live_query_encode.py
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import numpy as np

class query_encode:
    """
    Encodes queries using the MedCPT Query Encoder.
    """

    def __init__(self, model_name="ncbi/MedCPT-Query-Encoder", device=None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def encode(self, queries: list, batch_size: int = 100) -> np.ndarray:
        """
        Encode a list of query strings into embeddings.

        Args:
            queries: list of query strings
            batch_size: batch size for encoding

        Returns:
            np.ndarray of shape [num_queries, embedding_dim]
        """
        embeddings = []

        for i in tqdm(range(0, len(queries), batch_size), desc="Encoding queries"):
            batch = queries[i:i+batch_size]
            with torch.no_grad():
                encoded = self.tokenizer(
                    batch,
                    truncation=True,
                    padding=True,
                    return_tensors="pt",
                    max_length=192
                )
                encoded = {k: v.to(self.device) for k, v in encoded.items()}
                out = self.model(**encoded).last_hidden_state[:, 0, :]  # CLS token
                embeddings.extend(out.cpu().numpy())

        return np.vstack(embeddings)