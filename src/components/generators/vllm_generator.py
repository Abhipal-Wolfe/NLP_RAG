"""vLLM-based generator implementation"""

from typing import List, Optional, Dict, Any
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from ...core.document import Document

from ...core.interfaces import Generator, GenerationResult
from ...core.registry import register_component


@register_component("generator", "VLLMGenerator")
class VLLMGenerator(Generator):
    """
    General-purpose generator using vLLM for efficient inference.

    Supports any HuggingFace model with batch inference and GPU optimization.
    """

    def __init__(
        self,
        model_path: str,
        max_tokens: int = 200,
        temperature: float = 0.0,
        dtype: str = "half",
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.7,
        enforce_eager: bool = True,
        download_dir: Optional[str] = None,
        use_few_shot: bool = True,
        dataset_name: str = "med_qa",
        few_shot_examples: Optional[str] = None
    ):
        """
        Args:
            model_path: HuggingFace model path
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            dtype: Model dtype (half, float, etc.)
            tensor_parallel_size: Number of GPUs for tensor parallelism
            gpu_memory_utilization: GPU memory fraction to use
            enforce_eager: Disable CUDA graphs
            download_dir: Model download directory
            use_few_shot: Whether to use few-shot examples
            dataset_name: Dataset name for few-shot examples
            few_shot_examples: Custom few-shot examples string
        """
        self.model = LLM(
            model=model_path,
            download_dir=download_dir,
            dtype=dtype,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            enforce_eager=enforce_eager,
            disable_log_stats=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.use_few_shot = use_few_shot
        self.dataset_name = dataset_name
        self.few_shot_examples = few_shot_examples
        self.model_path = model_path

    def generate(
        self,
        query: str,
        context: Optional[List[Document]] = None,
        **kwargs
    ) -> GenerationResult:
        """
        Generate answer for a single query.

        Args:
            query: Question or instruction
            context: Retrieved documents (optional)
            **kwargs: Additional arguments

        Returns:
            GenerationResult with answer
        """
        results = self.generate_batch([query], [context] if context else None, **kwargs)
        return results[0]

    def generate_batch(
        self,
        queries: List[str],
        contexts: Optional[List[List[Document]]] = None,
        **kwargs
    ) -> List[GenerationResult]:
        """
        Generate answers for a batch of queries.

        Args:
            queries: List of questions
            contexts: List of context documents per query (optional)
            **kwargs: Additional arguments

        Returns:
            List of GenerationResult
        """
        # Format prompts
        prompts = []
        for i, query in enumerate(queries):
            context_docs = contexts[i] if contexts and i < len(contexts) else None
            prompt = self._format_prompt(query, context_docs)
            prompts.append(prompt)

        # Generate
        max_tokens = kwargs.get("max_tokens", self.max_tokens)
        temperature = kwargs.get("temperature", self.temperature)

        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=1.0,
            max_tokens=max_tokens,
            skip_special_tokens=False
        )

        results = self.model.generate(prompts, sampling_params, use_tqdm=False)

        # Convert to GenerationResult
        return [
            GenerationResult(
                answer=self._postprocess(pred.outputs[0].text),
                metadata={
                    "model": self.model_path,
                    "temperature": temperature,
                    "max_tokens": max_tokens
                }
            )
            for pred in results
        ]

    def _format_prompt(self, query: str, context: Optional[List[Document]] = None) -> str:
        """Format prompt with optional context and few-shot examples"""
        prompt_parts = []

        # Add few-shot examples if enabled
        if self.use_few_shot and self.few_shot_examples:
            prompt_parts.append(f"### Examples:\n{self.few_shot_examples}\n")

        # Add context if provided
        if context:
            context_text = "\n\n".join([doc.page_content for doc in context])
            prompt_parts.append(f"### Context:\n{context_text}\n")

        # Add query
        prompt_parts.append(f"### Question:\n{query}\n\n### Answer:\n")

        return "\n".join(prompt_parts)

    def _postprocess(self, answer: str) -> str:
        """Clean up generated answer"""
        # Remove special tokens
        special_tokens = ["</s>", "<|endoftext|>", "<pad>"]
        for token in special_tokens:
            answer = answer.replace(token, "")

        return answer.strip()
