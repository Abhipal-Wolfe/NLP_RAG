"""Abstract base classes for the modular RAG evaluation framework

These interfaces define the contract for swappable components:
- Retriever: Retrieves relevant documents
- Reranker: Reranks retrieved documents
- Generator: Generates answers from context
- Evaluator: Evaluates predictions
- Pipeline: Orchestrates the full RAG flow
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from .document import Document


@dataclass
class RetrievalResult:
    """Results from document retrieval"""
    documents: List[Document]
    scores: Optional[List[float]] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class RerankResult:
    """Results from document reranking"""
    documents: List[Document]
    scores: List[float]
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class GenerationResult:
    """Results from answer generation"""
    answer: str
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class EvaluationResult:
    """Results from evaluation"""
    metrics: Dict[str, Any]
    predictions: List[str]
    ground_truths: List[str]
    metadata: Optional[Dict[str, Any]] = None


class Retriever(ABC):
    """Abstract retriever interface"""

    @abstractmethod
    def retrieve(self, query: str, k: int = 5, **kwargs) -> RetrievalResult:
        """
        Retrieve top-k documents for a query.

        Args:
            query: Search query
            k: Number of documents to retrieve
            **kwargs: Additional retriever-specific arguments

        Returns:
            RetrievalResult with documents and scores
        """
        pass


class Reranker(ABC):
    """Abstract reranker interface"""

    @abstractmethod
    def rerank(
        self,
        query: str,
        documents: List[Document],
        top_k: Optional[int] = None,
        **kwargs
    ) -> RerankResult:
        """
        Rerank documents based on relevance to query.

        Args:
            query: Search query
            documents: Documents to rerank
            top_k: Number of top documents to return (None = all)
            **kwargs: Additional reranker-specific arguments

        Returns:
            RerankResult with reranked documents and scores
        """
        pass


class Generator(ABC):
    """Abstract generator interface"""

    @abstractmethod
    def generate(
        self,
        query: str,
        context: Optional[List[Document]] = None,
        **kwargs
    ) -> GenerationResult:
        """
        Generate an answer given a query and optional context.

        Args:
            query: Question or instruction
            context: Retrieved documents (optional)
            **kwargs: Additional generator-specific arguments

        Returns:
            GenerationResult with answer and metadata
        """
        pass

    @abstractmethod
    def generate_batch(
        self,
        queries: List[str],
        contexts: Optional[List[List[Document]]] = None,
        **kwargs
    ) -> List[GenerationResult]:
        """
        Generate answers for a batch of queries.

        Args:
            queries: List of questions or instructions
            contexts: List of context documents for each query (optional)
            **kwargs: Additional generator-specific arguments

        Returns:
            List of GenerationResult
        """
        pass


class Evaluator(ABC):
    """Abstract evaluator interface"""

    @abstractmethod
    def evaluate(
        self,
        predictions: List[str],
        ground_truths: List[str],
        **kwargs
    ) -> EvaluationResult:
        """
        Evaluate predictions against ground truths.

        Args:
            predictions: Model predictions
            ground_truths: Ground truth answers
            **kwargs: Additional evaluator-specific arguments

        Returns:
            EvaluationResult with metrics
        """
        pass


class Pipeline(ABC):
    """Abstract pipeline interface"""

    @abstractmethod
    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the full pipeline.

        Args:
            inputs: Input data (queries, ground truths, etc.)

        Returns:
            Dictionary with results and metrics
        """
        pass

    @abstractmethod
    def log_config(self) -> Dict[str, Any]:
        """
        Return pipeline configuration for reproducibility.

        Returns:
            Dictionary with configuration details
        """
        pass
