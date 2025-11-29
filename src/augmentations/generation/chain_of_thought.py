"""Chain of thought augmentation: Force step-by-step reasoning"""

from typing import Dict, Any, Optional, List


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
    # For now, just return empty kwargs
    # TODO: Implement CoT prompting
    return {}
