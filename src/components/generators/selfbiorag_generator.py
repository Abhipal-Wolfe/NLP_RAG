"""Self-BioRAG generator with reflection tokens"""

from typing import List, Optional, Dict, Any
from transformers import AutoTokenizer
try:
    from vllm import LLM, SamplingParams
except ImportError:
    # Mock for environments without vllm
    class LLM:
        def __init__(self, *args, **kwargs): pass
    class SamplingParams:
        def __init__(self, *args, **kwargs): pass

from ...core.document import Document
from ...prompts import FEW_SHOT_MEDQA

from ...core.interfaces import Generator, GenerationResult
from ...core.registry import register_component


@register_component("generator", "SelfBioRAGGenerator")
class SelfBioRAGGenerator(Generator):
    """
    Self-BioRAG generator that produces answers with reflection tokens.

    The model generates special tokens like [Retrieval], [Relevant], [Fully supported]
    as part of its output. Token log probabilities are used for scoring.
    """

    def __init__(
        self,
        model_path: str = "dmis-lab/selfbiorag_7b",
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
            model_path: HuggingFace model path (should be Self-BioRAG model)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            dtype: Model dtype
            tensor_parallel_size: Number of GPUs
            gpu_memory_utilization: GPU memory fraction
            enforce_eager: Disable CUDA graphs
            download_dir: Model download directory
            use_few_shot: Use few-shot examples
            dataset_name: Dataset for few-shot examples
            few_shot_examples: Custom few-shot examples
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
        # Use provided examples or default to FEW_SHOT_MEDQA
        self.few_shot_examples = few_shot_examples if few_shot_examples else FEW_SHOT_MEDQA
        self.model_path = model_path

        # Load reflection token IDs
        self.ret_tokens = {
            "[Retrieval]": self.tokenizer.convert_tokens_to_ids("[Retrieval]"),
            "[No Retrieval]": self.tokenizer.convert_tokens_to_ids("[No Retrieval]")
        }
        self.rel_tokens = {
            "[Relevant]": self.tokenizer.convert_tokens_to_ids("[Relevant]"),
            "[Irrelevant]": self.tokenizer.convert_tokens_to_ids("[Irrelevant]")
        }
        self.grd_tokens = {
            "[Fully supported]": self.tokenizer.convert_tokens_to_ids("[Fully supported]"),
            "[Partially supported]": self.tokenizer.convert_tokens_to_ids("[Partially supported]"),
            "[No support]": self.tokenizer.convert_tokens_to_ids("[No support]")
        }
        self.ut_tokens = {
            f"[Utility:{i}]": self.tokenizer.convert_tokens_to_ids(f"[Utility:{i}]")
            for i in range(1, 6)
        }

    def generate(
        self,
        query: str,
        context: Optional[List[Document]] = None,
        **kwargs
    ) -> GenerationResult:
        """Generate answer with reflection tokens"""
        results = self.generate_batch([query], [context] if context else None, **kwargs)
        return results[0]

    def generate_batch(
        self,
        queries: List[str],
        contexts: Optional[List[List[Document]]] = None,
        **kwargs
    ) -> List[GenerationResult]:
        """Generate answers with reflection tokens for batch"""
        prompts = []
        for i, query in enumerate(queries):
            context_docs = contexts[i] if contexts and i < len(contexts) else None
            prompt = self._format_prompt(query, context_docs)
            prompts.append(prompt)

        max_tokens = kwargs.get("max_tokens", self.max_tokens)
        temperature = kwargs.get("temperature", self.temperature)

        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=1.0,
            max_tokens=max_tokens,
            skip_special_tokens=False
        )

        results = self.model.generate(prompts, sampling_params, use_tqdm=False)

        return [
            GenerationResult(
                answer=self._postprocess(pred.outputs[0].text),
                metadata={
                    "model": self.model_path,
                    "raw_output": pred.outputs[0].text,
                    "token_ids": pred.outputs[0].token_ids,
                    "logprobs": pred.outputs[0].logprobs
                }
            )
            for pred in results
        ]

    def generate_with_logprobs(
        self,
        prompts: List[str],
        max_tokens: Optional[int] = None,
        logprobs: int = 20
    ) -> List[Dict]:
        """
        Generate with token log probabilities for reflection token extraction.

        Returns:
            List of dicts with text, token_ids, logprobs, cumulative_logprob
        """
        sampling_params = SamplingParams(
            temperature=self.temperature,
            top_p=1.0,
            max_tokens=max_tokens or self.max_tokens,
            logprobs=logprobs,
            skip_special_tokens=False
        )

        results = self.model.generate(prompts, sampling_params, use_tqdm=False)

        return [
            {
                "text": pred.outputs[0].text,
                "token_ids": pred.outputs[0].token_ids,
                "logprobs": pred.outputs[0].logprobs,
                "cumulative_logprob": pred.outputs[0].cumulative_logprob
            }
            for pred in results
        ]

    def _format_prompt(self, query: str, context: Optional[List[Document]] = None) -> str:
        """
        Format prompt for Self-BioRAG:
        
        Structure:
        1. Few-shot examples
        3. ## Question: 
        4. ## Options:
        5. ## Response:
        """
        prompt_parts = []

        # Add few-shot examples (FEW_SHOT_MEDQA format)
        if self.use_few_shot and self.few_shot_examples:
            prompt_parts.append(self.few_shot_examples)


        # Add question header, then query (which already includes ## Options:)
        prompt_parts.append(f"\n## Question:\n{query}")

        # Add response section
        prompt_parts.append(f"\n## Response:\n")

        # Add context with retrieval token if provided
        # Truncate documents to fit within model's 4096 token context limit
        # Reserve: ~1500 tokens for few-shot, ~500 for query, ~500 for generation
        # Remaining: ~1500 tokens for documents (~500 tokens each with top_k=3)
        MAX_DOC_CHARS_PER_DOC = 2500  # ~625 tokens per document
        MAX_TOTAL_CONTEXT_CHARS = 7000  # ~1750 tokens total for all documents
        
        if context:
            paragraphs = []
            total_chars = 0
            
            for doc in context:
                title = doc.metadata.get('title', '')
                content = doc.page_content
                
                # Truncate individual document if too long
                if len(content) > MAX_DOC_CHARS_PER_DOC:
                    content = content[:MAX_DOC_CHARS_PER_DOC] + "... [truncated]"
                
                # Check if adding this document would exceed total limit
                doc_text = f"{title}\n{content}" if title else content
                if total_chars + len(doc_text) > MAX_TOTAL_CONTEXT_CHARS:
                    # Skip remaining documents to avoid exceeding limit
                    break
                
                total_chars += len(doc_text)
                
                # Include title on first line if available
                if title:
                    paragraphs.append(f"[Retrieval]<paragraph>{title}\n{content}</paragraph>")
                else:
                    paragraphs.append(f"[Retrieval]<paragraph>{content}</paragraph>")
            
            context_text = "\n\n".join(paragraphs)
            prompt_parts.append(context_text)
        
        return "".join(prompt_parts)

    def _postprocess(self, answer: str) -> str:
        """Remove reflection tokens and special tokens from answer"""
        tokens_to_remove = [
            "[Retrieval]", "[No Retrieval]", "[Relevant]", "[Irrelevant]",
            "[Fully supported]", "[Partially supported]", "[No support]",
            *[f"[Utility:{i}]" for i in range(1, 6)],
            "</s>", "<|endoftext|>", "\n"
        ]
        for token in tokens_to_remove:
            answer = answer.replace(token, " ")

        return " ".join(answer.split()).strip()

    def get_reflection_tokens(self) -> Dict[str, Dict[str, int]]:
        """Get reflection token IDs"""
        return {
            "ret_tokens": self.ret_tokens,
            "rel_tokens": self.rel_tokens,
            "grd_tokens": self.grd_tokens,
            "ut_tokens": self.ut_tokens
        }
