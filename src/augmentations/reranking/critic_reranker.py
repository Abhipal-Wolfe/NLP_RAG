"""Critic reranker augmentation: Use LLM reflection tokens to score documents (Self-BioRAG)"""

from typing import List, Dict, Any, Tuple
import numpy as np
from ...core.document import Document
from ...prompts import format_critic_prompt


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
    verbose = config.get("verbose", False)
    idx = pipeline_context.get("idx", 0)

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
    for doc_idx, doc in enumerate(documents):
        title = doc.metadata.get("title", "")
        content = doc.page_content
        
        # Use standard format from prompts.py
        prompt = format_critic_prompt(query, content, title)
        prompts.append(prompt)
        
        if verbose and idx == 0:
            print(f"\n  --- DOCUMENT {doc_idx + 1} ---")
            print(f"  Title: {title if title else 'No title'}")
            print(f"  FAISS Score: {doc.metadata.get('score', 'N/A'):.4f}")
            print(f"  Content length: {len(content)} chars")
            print(f"  Content preview: {content[:200]}...")
            
            if doc_idx == 0:
                print(f"\n  --- CRITIC PROMPT EXAMPLE (Doc 1 only) ---")
                print(prompt)
                print("  " + "="*80)

    # Generate with logprobs
    if verbose and idx == 0:
        print(f"\n  --- GENERATING REFLECTION TOKENS FOR {len(documents)} DOCUMENTS ---")
    
    results = generator.generate_with_logprobs(prompts, max_tokens=100)  # Increased to ensure utility tokens aren't cut off
    
    if verbose and idx == 0:
        print(f"  Generation complete. Extracting reflection token probabilities...")

    # Score each document and extract individual components
    doc_scores = []
    doc_score_details = []
    
    for doc_idx, result in enumerate(results):
        pred_token_ids = result.get("token_ids", [])
        pred_log_probs = result.get("logprobs", [])
        
        # Extract individual scores
        rel_score = _extract_relevance_score(pred_log_probs, rel_tokens)
        grd_score = _extract_groundedness_score(pred_token_ids, pred_log_probs, grd_tokens)
        ut_score = _extract_utility_score(pred_token_ids, pred_log_probs, ut_tokens)
        
        # Compute final score
        final_score = w_rel * rel_score + w_sup * grd_score + w_use * ut_score
        
        doc_scores.append(final_score)
        doc_score_details.append({
            "relevance": rel_score,
            "groundedness": grd_score,
            "utility": ut_score,
            "final": final_score
        })
        
        # Show individual document scores
        if verbose and idx == 0:
            model_output = result.get('text', 'N/A')
            print(f"\n  --- DOCUMENT {doc_idx + 1} SCORING ---")
            print(f"  Model output: {model_output}")
            print(f"  Scores:")
            print(f"    Relevance:    {rel_score:.4f}")
            print(f"    Groundedness: {grd_score:.4f}")
            print(f"    Utility:      {ut_score:.4f}")
            print(f"    Final Score:  {final_score:.4f} (= {w_rel}*{rel_score:.3f} + {w_sup}*{grd_score:.3f} + {w_use}*{ut_score:.3f})")

    # Sort by score (descending)
    sorted_indices = np.argsort(doc_scores)[::-1]
    
    # Show ranking changes
    if verbose and idx == 0:
        print(f"\n  --- RANKING COMPARISON ---")
        print(f"  Original Order → Reranked Order (by critic score)")
        for new_rank, orig_idx in enumerate(sorted_indices[:top_k], 1):
            print(f"    Doc {orig_idx + 1} (FAISS: {documents[orig_idx].metadata.get('score', 'N/A'):.4f}) "
                  f"→ Rank {new_rank} (Critic: {doc_scores[orig_idx]:.4f})")
    
    reranked_docs = []
    
    for idx_in_sorted in sorted_indices[:top_k]:
        doc = documents[idx_in_sorted]
        details = doc_score_details[idx_in_sorted]
        
        # Add reranking scores to document metadata
        doc.metadata['rerank_score'] = details['final']
        doc.metadata['relevance_score'] = details['relevance']
        doc.metadata['support_score'] = details['groundedness']
        doc.metadata['utility_score'] = details['utility']
        
        reranked_docs.append(doc)
    
    reranked_scores = [doc_scores[i] for i in sorted_indices[:top_k]]

    return reranked_docs, {
        "reranked_scores": reranked_scores,
        "top_k": len(reranked_docs),
        "original_count": len(documents),
        "avg_relevance": np.mean([d["relevance"] for d in doc_score_details]),
        "avg_groundedness": np.mean([d["groundedness"] for d in doc_score_details]),
        "avg_utility": np.mean([d["utility"] for d in doc_score_details])
    }


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
