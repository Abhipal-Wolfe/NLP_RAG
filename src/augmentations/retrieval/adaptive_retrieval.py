"""Adaptive retrieval augmentation: Decide if retrieval is needed (Self-BioRAG)"""

from typing import List, Dict, Any, Tuple
import numpy as np


def adaptive_retrieval_augmentation(
    queries: List[str],
    metadata: Dict[str, Any],
    pipeline_context: Dict[str, Any],
    threshold: float = 0.5
) -> Tuple[bool, Dict[str, Any]]:
    """
    Decide if retrieval is needed using reflection tokens (Self-BioRAG).

    Args:
        queries: Input queries
        metadata: Query metadata
        pipeline_context: Context with generator and retriever
        threshold: Probability threshold for retrieval decision

    Returns:
        (should_retrieve, decision_info)

    How it works:
        1. Generate with special [Retrieval] / [No Retrieval] tokens
        2. Extract token log probabilities
        3. Compare probabilities to decide if retrieval is needed
    """
    generator = pipeline_context.get("generator")
    config = pipeline_context.get("config", {})
    verbose = config.get("verbose", False)
    idx = pipeline_context.get("idx", 0)

    # Check if generator supports reflection tokens
    if not hasattr(generator, "get_reflection_tokens"):
        # Always retrieve if generator doesn't support adaptive retrieval
        return True, {"mode": "always_retrieve", "reason": "no_reflection_tokens"}

    # Get reflection tokens
    tokens = generator.get_reflection_tokens()
    ret_tokens = tokens.get("ret_tokens", {})

    if not ret_tokens:
        return True, {"mode": "always_retrieve", "reason": "no_retrieval_tokens"}

    # Format query for retrieval decision
    query = queries[0]
    formatted_query = f"### Instruction:\n{query}\n\n### Response:\n"

    if verbose and idx == 0:
        print(f"\n  --- ADAPTIVE RETRIEVAL PROMPT ---")
        print(f"  {formatted_query}")
        print(f"  Length: {len(formatted_query)} chars")

    # Generate with logprobs
    results = generator.generate_with_logprobs([formatted_query], max_tokens=20)
    result = results[0]

    if verbose and idx == 0:
        print(f"\n  --- MODEL OUTPUT (first 20 tokens) ---")
        print(f"  Raw text: {result.get('text', 'N/A')}")
        print(f"  Token IDs: {result.get('token_ids', [])[:10]}")  # First 10
        
        # Show logprobs for retrieval tokens
        pred_log_probs = result.get("logprobs", [])
        if pred_log_probs:
            first_token_probs = pred_log_probs[0]
            print(f"\n  --- RETRIEVAL TOKEN LOGPROBS (first position) ---")
            for tok_name, tok_id in ret_tokens.items():
                if tok_id in first_token_probs:
                    logprob_obj = first_token_probs[tok_id]
                    if hasattr(logprob_obj, 'logprob'):
                        logprob_val = float(logprob_obj.logprob)
                    elif isinstance(logprob_obj, (int, float)):
                        logprob_val = float(logprob_obj)
                    else:
                        logprob_val = -100
                    prob = np.exp(logprob_val) if logprob_val > -100 else 0.0
                    print(f"  {tok_name}: logprob={logprob_val:.4f}, prob={prob:.4f}")
                else:
                    print(f"  {tok_name}: NOT FOUND in first position")

    # Extract retrieval token probabilities from first position
    pred_log_probs = result.get("logprobs", [])
    if not pred_log_probs:
        return True, {"mode": "always_retrieve", "reason": "no_logprobs"}

    first_token_probs = pred_log_probs[0]

    # Extract scores
    score_dict = {}
    for tok, token_id in ret_tokens.items():
        if token_id in first_token_probs:
            logprob_obj = first_token_probs[token_id]
            if hasattr(logprob_obj, 'logprob'):
                score_dict[tok] = float(logprob_obj.logprob)
            elif isinstance(logprob_obj, (int, float)):
                score_dict[tok] = float(logprob_obj)
            else:
                score_dict[tok] = -100
        else:
            score_dict[tok] = -100

    ret_prob = score_dict.get("[Retrieval]", -100)
    no_ret_prob = score_dict.get("[No Retrieval]", -100)

    # Decide based on probability ratio
    if ret_prob != -100 and no_ret_prob != -100:
        ret_prob_exp = np.exp(ret_prob)
        no_ret_prob_exp = np.exp(no_ret_prob)
        total_prob = ret_prob_exp + no_ret_prob_exp

        if total_prob > 0:
            ratio = ret_prob_exp / total_prob
            should_retrieve = ratio > threshold
            
            if verbose and idx == 0:
                print(f"\n  --- DECISION CALCULATION ---")
                print(f"  [Retrieval] prob: {ret_prob_exp:.4f}")
                print(f"  [No Retrieval] prob: {no_ret_prob_exp:.4f}")
                print(f"  Ratio: {ratio:.4f} (threshold: {threshold})")
                print(f"  Decision: {'RETRIEVE' if should_retrieve else 'NO RETRIEVAL'}")
        else:
            should_retrieve = False
    else:
        # Fallback: always retrieve if tokens not found
        should_retrieve = True
        
        if verbose and idx == 0:
            print(f"\n  --- DECISION FALLBACK ---")
            print(f"  Tokens not found in expected position, defaulting to RETRIEVE")

    return should_retrieve, {
        "mode": "adaptive",
        "retrieval_logprob": ret_prob,
        "no_retrieval_logprob": no_ret_prob,
        "retrieval_prob": np.exp(ret_prob) if ret_prob > -100 else 0.0,
        "no_retrieval_prob": np.exp(no_ret_prob) if no_ret_prob > -100 else 0.0,
        "threshold": threshold,
        "should_retrieve": should_retrieve
    }


def create_adaptive_retrieval_augmentation(threshold: float = 0.5):
    """
    Factory function to create adaptive retrieval augmentation with custom threshold.

    Args:
        threshold: Probability threshold for retrieval decision

    Returns:
        Augmentation function with threshold baked in
    """
    def augmentation(queries, metadata, pipeline_context):
        return adaptive_retrieval_augmentation(
            queries, metadata, pipeline_context, threshold=threshold
        )

    augmentation.__name__ = f"adaptive_retrieval_{threshold}"
    return augmentation
