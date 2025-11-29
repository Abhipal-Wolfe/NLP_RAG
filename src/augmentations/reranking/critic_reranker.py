"""Critic reranker augmentation: Use LLM reflection tokens to score documents (Self-BioRAG)"""

from typing import List, Dict, Any, Tuple
import numpy as np
from ...core.document import Document


def critic_reranker_augmentation(
    query: str,
    documents: List[Document],
    metadata: Dict[str, Any],
    pipeline_context: Dict[str, Any],
    top_k: int = 3,
    w_rel: float = 1.0,
    w_sup: float = 1.0,
    w_use: float = 0.5
) -> Tuple[List[Document], List[float]]:
    """
    Rerank documents using LLM-generated reflection tokens (Self-BioRAG).

    Uses reflection tokens to score documents:
    - [Relevant] / [Irrelevant]: Relevance to query
    - [Fully supported] / [Partially supported]: Groundedness
    - [Utility:1-5]: Utility for answering

    Args:
        query: Query string
        documents: Documents to rerank
        metadata: Query metadata
        pipeline_context: Context with generator and config
        top_k: Number of top documents to return
        w_rel: Weight for relevance score
        w_sup: Weight for groundedness score
        w_use: Weight for utility score

    Returns:
        (reranked_documents, scores)
    """
    generator = pipeline_context.get("generator")
    config = pipeline_context.get("config", {})

    # Check if generator supports reflection tokens
    if not hasattr(generator, "get_reflection_tokens"):
        # Return documents as-is if generator doesn't support reflection
        return documents[:top_k], [1.0] * min(top_k, len(documents))

    # Get reflection tokens
    tokens = generator.get_reflection_tokens()
    rel_tokens = tokens.get("rel_tokens", {})
    grd_tokens = tokens.get("grd_tokens", {})
    ut_tokens = tokens.get("ut_tokens", {})

    # Generate prompts for each document
    prompts = []
    for doc in documents:
        title = doc.metadata.get("title", "")
        content = doc.page_content
        paragraph = f"{title}\n{content}" if title else content
        prompt = f"{query}[Retrieval]<paragraph>{paragraph}</paragraph>\n\n### Response:\n"
        prompts.append(prompt)

    # Generate with logprobs
    results = generator.generate_with_logprobs(prompts)

    # Score each document
    doc_scores = []
    for result in results:
        score = _compute_score(
            result,
            rel_tokens,
            grd_tokens,
            ut_tokens,
            w_rel,
            w_sup,
            w_use
        )
        doc_scores.append(score)

    # Sort by score (descending)
    sorted_indices = np.argsort(doc_scores)[::-1]
    reranked_docs = [documents[i] for i in sorted_indices[:top_k]]
    reranked_scores = [doc_scores[i] for i in sorted_indices[:top_k]]

    return reranked_docs, reranked_scores


def _compute_score(
    result: Dict[str, Any],
    rel_tokens: Dict[str, int],
    grd_tokens: Dict[str, int],
    ut_tokens: Dict[str, int],
    w_rel: float,
    w_sup: float,
    w_use: float
) -> float:
    """Compute weighted score from reflection tokens"""
    pred_token_ids = result.get("token_ids", [])
    pred_log_probs = result.get("logprobs", [])

    # Extract individual scores
    rel_score = _extract_relevance_score(pred_log_probs, rel_tokens)
    grd_score = _extract_groundedness_score(pred_token_ids, pred_log_probs, grd_tokens)
    ut_score = _extract_utility_score(pred_token_ids, pred_log_probs, ut_tokens)

    # Weighted sum
    final_score = w_rel * rel_score + w_sup * grd_score + w_use * ut_score

    return final_score


def _extract_relevance_score(pred_log_probs: List[Dict], rel_tokens: Dict[str, int]) -> float:
    """Extract [Relevant] vs [Irrelevant] score"""
    if not pred_log_probs:
        return 0.0

    first_token_probs = pred_log_probs[0]
    score_dict = {}

    for tok, token_id in rel_tokens.items():
        if token_id in first_token_probs:
            logprob_obj = first_token_probs[token_id]
            if hasattr(logprob_obj, 'logprob'):
                score_dict[tok] = np.exp(float(logprob_obj.logprob))
            elif isinstance(logprob_obj, (int, float)):
                score_dict[tok] = np.exp(float(logprob_obj))
            else:
                score_dict[tok] = 0.0
        else:
            score_dict[tok] = 0.0

    total = sum(score_dict.values())
    return score_dict.get("[Relevant]", 0) / total if total > 0 else 0.0


def _extract_groundedness_score(
    pred_token_ids: List[int],
    pred_log_probs: List[Dict],
    grd_tokens: Dict[str, int]
) -> float:
    """Extract [Fully supported]/[Partially supported] score"""
    if not grd_tokens:
        return 0.0

    score_dict = _find_and_extract_token_probs(pred_token_ids, pred_log_probs, grd_tokens)

    if len(score_dict) != 3:
        return 0.0

    total = sum(score_dict.values())
    if total == 0:
        return 0.0

    return (
        score_dict.get("[Fully supported]", 0) / total +
        0.5 * score_dict.get("[Partially supported]", 0) / total
    )


def _extract_utility_score(
    pred_token_ids: List[int],
    pred_log_probs: List[Dict],
    ut_tokens: Dict[str, int]
) -> float:
    """Extract [Utility:1-5] score"""
    if not ut_tokens:
        return 0.0

    score_dict = _find_and_extract_token_probs(pred_token_ids, pred_log_probs, ut_tokens)

    if len(score_dict) != 5:
        return 0.0

    total = sum(score_dict.values())
    if total == 0:
        return 0.0

    # Map to [-1, 1] scale
    weights = [-1, -0.5, 0, 0.5, 1]
    return sum(
        weights[i] * score_dict.get(f"[Utility:{i+1}]", 0) / total
        for i in range(5)
    )


def _find_and_extract_token_probs(
    pred_token_ids: List[int],
    pred_log_probs: List[Dict],
    token_dict: Dict[str, int]
) -> Dict[str, float]:
    """Find where special tokens appear and extract probabilities"""
    for tok_idx, token_id in enumerate(pred_token_ids):
        if token_id in token_dict.values() and tok_idx < len(pred_log_probs):
            token_probs = pred_log_probs[tok_idx]
            result = {}

            for tok, tok_id in token_dict.items():
                if tok_id in token_probs:
                    logprob_obj = token_probs[tok_id]
                    if hasattr(logprob_obj, 'logprob'):
                        result[tok] = np.exp(float(logprob_obj.logprob))
                    elif isinstance(logprob_obj, (int, float)):
                        result[tok] = np.exp(float(logprob_obj))
                    else:
                        result[tok] = 0.0
                else:
                    result[tok] = 0.0

            return result

    return {}


def create_critic_reranker_augmentation(
    top_k: int = 3,
    w_rel: float = 1.0,
    w_sup: float = 1.0,
    w_use: float = 0.5
):
    """
    Factory function to create critic reranker with custom parameters.

    Args:
        top_k: Number of documents to return
        w_rel: Weight for relevance
        w_sup: Weight for groundedness
        w_use: Weight for utility

    Returns:
        Augmentation function with parameters baked in
    """
    def augmentation(query, documents, metadata, pipeline_context):
        return critic_reranker_augmentation(
            query, documents, metadata, pipeline_context,
            top_k=top_k, w_rel=w_rel, w_sup=w_sup, w_use=w_use
        )

    augmentation.__name__ = f"critic_reranker_top{top_k}"
    return augmentation
