"""Self-reflection augmentation: Validate and refine answers (Self-BioRAG)"""

from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import re


def self_reflection_augmentation(
    query: str,
    answer: str,
    context: Optional[List[Any]],
    metadata: Dict[str, Any],
    pipeline_context: Dict[str, Any],
    max_iterations: int = 3
) -> Tuple[str, Dict[str, Any]]:
    """
    Self-critique and refine answer using [Utility] tokens (Self-BioRAG).

    Args:
        query: Original query
        answer: Generated answer
        context: Retrieved documents
        metadata: Query metadata
        pipeline_context: Context with generator and config
        max_iterations: Maximum refinement iterations

    Returns:
        (refined_answer, reflection_info)

    How it works:
        1. Generate with [Utility:1-5] tokens
        2. Check utility score (1=bad, 5=great)
        3. If score < 4, regenerate up to max_iterations
    """
    generator = pipeline_context.get("generator")

    # Check if generator supports reflection tokens
    if not hasattr(generator, "get_reflection_tokens"):
        return answer, {"iterations": 0, "refined": False, "reason": "no_reflection_support"}

    # Get reflection tokens
    tokens = generator.get_reflection_tokens()
    ut_tokens = tokens.get("ut_tokens", {})

    if not ut_tokens:
        return answer, {"iterations": 0, "refined": False, "reason": "no_utility_tokens"}

    verbose = pipeline_context.get("config", {}).get("verbose", False)
    idx = pipeline_context.get("idx", 0)
    
    # Try to get utility from the ORIGINAL answer generation (if available)
    # Self-BioRAG generates utility tokens as part of the answer, not in a separate step
    generation_result = pipeline_context.get("generation_result")
    if generation_result and hasattr(generation_result, 'metadata'):
        raw_token_ids = generation_result.metadata.get('token_ids', [])
        raw_logprobs = generation_result.metadata.get('logprobs', [])
        
        if verbose and idx == 0:
            print(f"\n  --- EXTRACTING UTILITY FROM ORIGINAL GENERATION ---")
            print(f"  Raw output: {generation_result.metadata.get('raw_output', 'N/A')}...")
        
        # Extract utility from original generation
        utility_score = _extract_utility_score(raw_token_ids, raw_logprobs, ut_tokens)
        
        if verbose and idx == 0:
            print(f"\n  --- UTILITY TOKEN SEARCH IN ORIGINAL OUTPUT ---")
            print(f"  Total tokens generated: {len(raw_token_ids)}")
            print(f"  Searching for [Utility:1-5] tokens...")
            
            found_token = None
            for tok_name, tok_id in ut_tokens.items():
                found = tok_id in raw_token_ids
                if found:
                    pos = raw_token_ids.index(tok_id)
                    found_token = tok_name
                    print(f"  {tok_name} (ID={tok_id}): ✓ FOUND at position {pos}")
                else:
                    print(f"  {tok_name} (ID={tok_id}): ✗ NOT FOUND")
            
            if utility_score >= 0:
                print(f"\n  ✓ Extracted utility score: {utility_score:.1f} from token {found_token}")
            else:
                print(f"\n  ✗ Utility score: -1 (extraction failed)")
                print(f"  This could mean token parsing failed or logprobs unavailable.")
        
        # If we found utility in original answer, return immediately
        if utility_score >= 0:
            return answer, {
                "iterations": 1,
                "refined": False,
                "utility_scores": [utility_score],
                "final_utility": utility_score,
                "stopped_reason": "extracted_from_original",
                "source": "original_generation"
            }
    
    # Fallback: Try regeneration-based reflection (if utility not in original)
    if verbose and idx == 0:
        print(f"\n  ⚠️  Utility not found in original generation, trying regeneration approach...")
    
    best_answer = answer
    best_utility = -1
    utility_scores = []

    for iteration in range(max_iterations):
        # Regenerate answer with context
        from ...core.document import Document
        answer = generator.generate(query, context=context).answer
        
        # Try to extract utility from this generation
        generation_result = pipeline_context.get("generation_result")
        if generation_result and hasattr(generation_result, 'metadata'):
            raw_token_ids = generation_result.metadata.get('token_ids', [])
            raw_logprobs = generation_result.metadata.get('logprobs', [])
            utility_score = _extract_utility_score(raw_token_ids, raw_logprobs, ut_tokens)
        else:
            utility_score = -1
        
        utility_scores.append(utility_score)

        # Update best answer if this is better
        if utility_score > best_utility:
            best_utility = utility_score
            best_answer = answer

        # If utility is high enough (>=4), stop
        if utility_score >= 4:
            return best_answer, {
                "iterations": iteration + 1,
                "refined": iteration > 0,
                "utility_scores": utility_scores,
                "final_utility": utility_score,
                "stopped_reason": "high_utility"
            }

        # Regenerate answer if utility is low and not last iteration
        if iteration < max_iterations - 1 and utility_score < 4:
            # Regenerate with context
            from ...core.document import Document
            answer = generator.generate(query, context=context).answer

    return best_answer, {
        "iterations": len(utility_scores),
        "refined": len(utility_scores) > 1,
        "utility_scores": utility_scores,
        "final_utility": best_utility,
        "stopped_reason": "max_iterations"
    }


def _extract_utility_score(
    pred_token_ids: List[int],
    pred_log_probs: List[Dict],
    ut_tokens: Dict[str, int]
) -> float:
    """
    Extract utility score (1-5) from [Utility:X] tokens.
    
    Simply checks if any utility token appears in the sequence and returns its value.
    
    Returns:
        Float score 1-5, or -1 if not found
    """
    if not ut_tokens or not pred_token_ids:
        return -1

    # Search for ANY utility token in the entire sequence
    for token_id in pred_token_ids:
        if token_id in ut_tokens.values():
            # Found a utility token! Determine which one (1-5)
            for tok_name, ut_tok_id in ut_tokens.items():
                if ut_tok_id == token_id:
                    # Extract the utility level from token name like "[Utility:4]"
                    # tok_name format: "[Utility:X]" where X is 1-5
                    match = re.search(r'\[Utility:(\d)\]', tok_name)
                    if match:
                        utility_level = int(match.group(1))
                        return float(utility_level)
    
    # No utility token found
    return -1


def create_self_reflection_augmentation(max_iterations: int = 2):
    """
    Factory function to create self-reflection augmentation with custom iterations.

    Args:
        max_iterations: Maximum refinement iterations

    Returns:
        Augmentation function with parameters baked in
    """
    def augmentation(query, answer, context, metadata, pipeline_context):
        return self_reflection_augmentation(
            query, answer, context, metadata, pipeline_context,
            max_iterations=max_iterations
        )

    augmentation.__name__ = f"self_reflection_iter{max_iterations}"
    return augmentation
