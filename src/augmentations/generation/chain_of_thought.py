"""Chain of thought augmentation: Force step-by-step reasoning"""

from typing import Dict, Any, Optional, List, Callable
from src.prompts import format_cot_prompt_with_options


def create_chain_of_thought_augmentation() -> Callable:
    """Factory for CoT augmentation"""
    return chain_of_thought_augmentation


def chain_of_thought_augmentation(
    query: str,
    context: Optional[List[Any]],
    metadata: Dict[str, Any],
    pipeline_context: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Force chain-of-thought reasoning during generation.

    Args:
        query: Query string
        context: Retrieved documents
        metadata: Query metadata
        pipeline_context: Context with generator and config

    Returns:
        kwargs to pass to generator (e.g., modified prompt)
    """
    # Get original item from context if available
    idx = pipeline_context.get("idx")
    ground_truths = pipeline_context.get("ground_truths")
    
    if idx is not None and ground_truths:
        item = ground_truths[idx]
        # Format using the CoT prompt template
        cot_prompt = format_cot_prompt_with_options(item)
        
        # If we have context, append it (optional, depending on if we want RAG + CoT)
        # For now, let's assume this replaces the query entirely
        return {"query": cot_prompt}
        
    return {}
