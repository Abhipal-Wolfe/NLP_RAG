"""Augmentation loader: Instantiate augmentations from config"""

from typing import Dict, Any, List, Callable
from .query_processing.multi_query import create_multi_query_augmentation
from .retrieval.adaptive_retrieval import create_adaptive_retrieval_augmentation
from .reranking.critic_reranker import create_critic_reranker_augmentation
from .reflection.self_reflection import create_self_reflection_augmentation


def load_augmentations(config: Dict[str, Any]) -> Dict[str, List[Callable]]:
    """
    Load and instantiate augmentations from config.
    
    Args:
        config: Full pipeline config with 'augmentations' section
        
    Returns:
        Dictionary mapping stage names to lists of augmentation functions
    """
    aug_config = config.get("augmentations", {})
    
    augmentations = {
        "query": [],
        "retrieval": [],
        "rerank": [],
        "generation": [],
        "reflection": []
    }
    
    # Load query processing augmentations
    for aug_spec in aug_config.get("query", []):
        if isinstance(aug_spec, dict):
            aug_name = list(aug_spec.keys())[0]
            aug_params = aug_spec.get(aug_name, {})
            
            if aug_name == "multi_query" or aug_name == "query_decomposition":
                max_sub_queries = aug_params.get("max_sub_queries", 4)
                augmentations["query"].append(
                    create_multi_query_augmentation(max_sub_queries=max_sub_queries)
                )
    
    # Load retrieval augmentations
    for aug_spec in aug_config.get("retrieval", []):
        if isinstance(aug_spec, dict):
            aug_name = list(aug_spec.keys())[0]
            aug_params = aug_spec.get(aug_name, {})
            
            if aug_name == "adaptive_retrieval":
                threshold = aug_params.get("threshold", 0.5)
                augmentations["retrieval"].append(
                    create_adaptive_retrieval_augmentation(threshold=threshold)
                )
    
    # Load rerank augmentations
    for aug_spec in aug_config.get("rerank", []):
        if isinstance(aug_spec, dict):
            aug_name = list(aug_spec.keys())[0]
            aug_params = aug_spec.get(aug_name, {})
            
            if aug_name == "critic_reranker":
                top_k = aug_params.get("top_k", 3)
                w_rel = aug_params.get("w_rel", 1.0)
                w_sup = aug_params.get("w_sup", 1.0)
                w_use = aug_params.get("w_use", 0.5)
                augmentations["rerank"].append(
                    create_critic_reranker_augmentation(
                        top_k=top_k, w_rel=w_rel, w_sup=w_sup, w_use=w_use
                    )
                )
    
    # Load reflection augmentations
    for aug_spec in aug_config.get("reflection", []):
        if isinstance(aug_spec, dict):
            aug_name = list(aug_spec.keys())[0]
            aug_params = aug_spec.get(aug_name, {})
            
            if aug_name == "self_reflection":
                max_iterations = aug_params.get("max_iterations", 3)
                augmentations["reflection"].append(
                    create_self_reflection_augmentation(max_iterations=max_iterations)
                )
    
    return augmentations


def print_loaded_augmentations(augmentations: Dict[str, List[Callable]]):
    """Print summary of loaded augmentations"""
    print("\n" + "="*100)
    print("LOADED AUGMENTATIONS")
    print("="*100)
    
    for stage, augs in augmentations.items():
        if augs:
            print(f"{stage.upper()} ({len(augs)}):")
            for aug in augs:
                aug_name = getattr(aug, '__name__', str(aug))
                print(f"  - {aug_name}")
    
    total = sum(len(augs) for augs in augmentations.values())
    if total == 0:
        print("No augmentations loaded (standard RAG mode)")
    
    print("="*100 + "\n")

