"""
Self-BioRAG: Retrieval-Augmented Generation with Self-Reflection for Biomedical QA

A clean, production-ready implementation of Self-BioRAG following the original paper.
"""

__version__ = "0.1.0"

from .generator_models import GeneratorModel, ScoringResult
from .retriever_models import BiomedicalRetriever, VectorStoreRetriever, PreloadedRetriever, RetrievalResult
from .chains import (
    Chain,
    BaselineChain,
    BaselineConfig,
    EvaluationResult,
    SelfBioRAGChain,
    SelfBioRAGConfig,
    SelfBioRAGEvaluationResult,
)
from .infrastructure import (
    SelfBioRAGFactory,
    load_reflection_tokens,
    normalize_answer,
    calculate_accuracy,
    load_jsonl,
    save_jsonl
)

__all__ = [
    # Models
    "GeneratorModel",
    "ScoringResult",
    # Retrievers
    "BiomedicalRetriever",
    "VectorStoreRetriever",
    "PreloadedRetriever",
    "RetrievalResult",
    # Chains
    "Chain",
    "BaselineChain",
    "BaselineConfig",
    "EvaluationResult",
    "SelfBioRAGChain",
    "SelfBioRAGConfig",
    "SelfBioRAGEvaluationResult",
    # Factory
    "SelfBioRAGFactory",
    # Utils
    "load_reflection_tokens",
    "normalize_answer",
    "calculate_accuracy",
    "load_jsonl",
    "save_jsonl",
]

