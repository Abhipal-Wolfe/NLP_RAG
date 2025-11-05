"""Model wrappers for Self-BioRAG components

Key: Self-BioRAG uses ONE model that generates answers AND reflection tokens.
No separate critic model during inference.
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


@dataclass
class ScoringResult:
    """Scoring result from reflection tokens"""
    relevance_score: float
    groundedness_score: float
    utility_score: float
    final_score: float


class GeneratorModel:
    """
    Self-BioRAG generator that produces answers with reflection tokens.
    
    The model generates special tokens like [Retrieval], [Relevant], [Fully supported]
    as part of its output. Token log probabilities are used for scoring.
    """
    
    def __init__(
        self,
        model_path: str,
        max_tokens: int = 200,
        temperature: float = 0.0,
        dtype: str = "half",
        tensor_parallel_size: int = 1,
        download_dir: Optional[str] = None,
        gpu_memory_utilization: float = 0.7,
        enforce_eager: bool = True
    ):
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
    
    def generate(self, prompts: List[str], max_tokens: Optional[int] = None) -> List[Dict]:
        """Basic generation without logprobs"""
        sampling_params = SamplingParams(
            temperature=self.temperature,
            top_p=1.0,
            max_tokens=max_tokens or self.max_tokens,
            skip_special_tokens=False
        )
        # Suppress vLLM's internal progress bar
        results = self.model.generate(prompts, sampling_params, use_tqdm=False)
        return [{"text": pred.outputs[0].text} for pred in results]
    
    def generate_with_logprobs(
        self,
        prompts: List[str],
        max_tokens: Optional[int] = None,
        logprobs: int = 20  # vLLM limit is 20 logprobs per token position
    ) -> List[Dict]:
        """
        Generate with token log probabilities for reflection token extraction.
        
        Returns list of dicts with:
            - text: generated text
            - token_ids: list of token IDs
            - logprobs: list of dicts mapping token_id -> log_prob per position
            - cumulative_logprob: total log prob of sequence
        """
        sampling_params = SamplingParams(
            temperature=self.temperature,
            top_p=1.0,
            max_tokens=max_tokens or self.max_tokens,
            logprobs=logprobs,
            skip_special_tokens=False
        )
        # Suppress vLLM's internal progress bar ("Processed prompts" log)
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


def load_reflection_tokens(tokenizer: AutoTokenizer) -> Dict[str, Dict[str, int]]:
    """
    Load reflection token IDs from tokenizer.
    
    Returns dict with keys: ret_tokens, rel_tokens, grd_tokens, ut_tokens
    """
    ret_tokens = {
        "[Retrieval]": tokenizer.convert_tokens_to_ids("[Retrieval]"),
        "[No Retrieval]": tokenizer.convert_tokens_to_ids("[No Retrieval]")
    }
    
    rel_tokens = {
        "[Relevant]": tokenizer.convert_tokens_to_ids("[Relevant]"),
        "[Irrelevant]": tokenizer.convert_tokens_to_ids("[Irrelevant]")
    }
    
    grd_tokens = {
        "[Fully supported]": tokenizer.convert_tokens_to_ids("[Fully supported]"),
        "[Partially supported]": tokenizer.convert_tokens_to_ids("[Partially supported]"),
        "[No support]": tokenizer.convert_tokens_to_ids("[No support]")
    }
    
    ut_tokens = {
        f"[Utility:{i}]": tokenizer.convert_tokens_to_ids(f"[Utility:{i}]")
        for i in range(1, 6)
    }
    
    return {
        "ret_tokens": ret_tokens,
        "rel_tokens": rel_tokens,
        "grd_tokens": grd_tokens,
        "ut_tokens": ut_tokens
    }
