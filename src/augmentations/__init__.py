"""Modular augmentations for RAG pipeline

Augmentations are functions that can be injected at different pipeline stages:
- Query Processing: Transform queries before retrieval
- Retrieval: Decide if/how to retrieve
- Reranking: Score and reorder documents
- Generation: Modify generation behavior
- Reflection: Post-process/validate answers
"""

from .query_processing import *
from .retrieval import *
from .reranking import *
from .generation import *
from .reflection import *

__all__ = [
    # Query processing
    "multi_query_augmentation",
    "reasoning_chain_augmentation",
    # Retrieval
    "adaptive_retrieval_augmentation",
    # Reranking
    "critic_reranker_augmentation",
    # Generation
    "chain_of_thought_augmentation",
    # Reflection
    "self_reflection_augmentation",
]
