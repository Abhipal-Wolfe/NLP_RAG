"""Mock generator for testing"""

from typing import List, Optional, Dict, Any
from ...core.document import Document
from ...core.interfaces import Generator, GenerationResult
from ...core.registry import register_component

@register_component("generator", "MockGenerator")
class MockGenerator(Generator):
    """
    Mock generator that returns dummy responses.
    Useful for testing pipeline logic without loading heavy models.
    """

    def __init__(
        self,
        model_path: str = "mock-model",
        max_tokens: int = 200,
        temperature: float = 0.0,
        **kwargs
    ):
        self.model_path = model_path
        self.max_tokens = max_tokens
        self.temperature = temperature

    def generate(
        self,
        query: str,
        context: Optional[List[Document]] = None,
        raw: bool = False,
        **kwargs
    ) -> GenerationResult:
        return GenerationResult(
            answer="Mock Answer: A. Vitamin B1 (Thiamine)",
            metadata={"model": "mock"}
        )

    def generate_batch(
        self,
        queries: List[str],
        contexts: Optional[List[List[Document]]] = None,
        raw: bool = False,
        **kwargs
    ) -> List[GenerationResult]:
        return [
            GenerationResult(
                answer="Mock Answer: A. Vitamin B1 (Thiamine)",
                metadata={"model": "mock"}
            ) for _ in queries
        ]

    def _format_prompt(self, query: str, context: Optional[List[Document]] = None) -> str:
        """Simple pass-through for prompt inspection"""
        return query
