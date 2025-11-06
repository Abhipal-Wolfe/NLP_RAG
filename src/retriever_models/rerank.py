import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import gc

# ---------------------------
# Combine multiple evidence sources per query
# ---------------------------
def combine_query_evidence(queries, *evidence_lists):
    """
    Combine multiple evidence lists per query.

    Args:
        queries (list): list of query strings
        *evidence_lists: any number of lists of evidences (each list: list of lists per query)

    Returns:
        q_evidence_pairs: list of list of [query, evidence] pairs per query
        combined_evidences: list of combined evidence strings per query
    """
    combined_evidences = [
        sum(evidences_per_list, [])  # concatenates evidence lists from each source
        for evidences_per_list in zip(*evidence_lists)
    ]

    q_evidence_pairs = [
        [[q, ev] for ev in combined_evidences[i]] for i, q in enumerate(queries)
    ]

    return q_evidence_pairs, combined_evidences


# ---------------------------
# Rerank all evidence per query at once
# ---------------------------
def rerank(q_4a_list, evidences_4, tokenizer, model, device):
    """
    Rerank evidence for each query using a cross-encoder.

    Args:
        q_4a_list: list of list of [query, evidence] pairs (one list per query)
        evidences_4: list of list of evidence strings (one list per query)
        tokenizer: preloaded AutoTokenizer
        model: preloaded AutoModelForSequenceClassification
        device: 'cuda' or 'cpu'

    Returns:
        sorted_evidence_list: top-10 evidences per query after reranking
    """
    sorted_evidence_list = []

    for q_pairs, ev_list in tqdm(zip(q_4a_list, evidences_4), total=len(q_4a_list), desc="Reranking queries"):
        with torch.no_grad():
            encoded = tokenizer(q_pairs, truncation=True, padding=True, return_tensors="pt", max_length=512)
            encoded = {k: v.to(device) for k, v in encoded.items()}
            logits = model(**encoded).logits.squeeze(dim=1)

        # Sort and pick top 10
        query_logits = logits.detach().cpu().tolist()
        top_indices = sorted(range(len(query_logits)), key=lambda k: query_logits[k], reverse=True)[:10]
        sorted_evidence_list.append([ev_list[i] for i in top_indices])

        del encoded, logits
        torch.cuda.empty_cache()
        gc.collect()

    return sorted_evidence_list


from sentence_transformers import util
import torch
import gc
from tqdm import tqdm

def rerank_sbert(q_4a_list, evidences_4, sbert_model, device, topk=10):
    """
    Fully batched SBERT reranking for small mini-batches (~160 embeddings).
    """
    sorted_evidence_list = []

    # 1. Extract queries
    query_texts = [q_pairs[0][0] for q_pairs in q_4a_list]

    # 2. Flatten evidences
    all_evidences = [ev for ev_list in evidences_4 for ev in ev_list]

    # 3. Encode queries and evidences at once
    query_embs = sbert_model.encode(query_texts, convert_to_tensor=True, device=device, show_progress_bar=False)
    evidence_embs = sbert_model.encode(all_evidences, convert_to_tensor=True, device=device, show_progress_bar=False)

    # 4. Compute cosine similarities per query
    start_idx = 0
    for i, ev_list in enumerate(evidences_4):
        end_idx = start_idx + len(ev_list)
        cos_scores = util.cos_sim(query_embs[i], evidence_embs[start_idx:end_idx]).squeeze(0)
        top_indices = torch.topk(cos_scores, k=min(topk, len(ev_list))).indices.tolist()
        sorted_evidence_list.append([ev_list[j] for j in top_indices])
        start_idx = end_idx

    # Cleanup
    del query_embs, evidence_embs, cos_scores, top_indices
    torch.cuda.empty_cache()
    gc.collect()

    return sorted_evidence_list