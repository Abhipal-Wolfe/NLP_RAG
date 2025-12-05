"""LLM-based critic reranker for Self-BioRAG"""

from typing import List, Optional, Dict, Any
from ...core.document import Document
import numpy as np

from ...core.interfaces import Reranker, RerankResult
from ...core.registry import register_component


@register_component("reranker", "LLMCriticReranker")
class LLMCriticReranker(Reranker):
    """
    Reranker that uses LLM-generated reflection tokens to score documents.

    Used in Self-BioRAG for scoring document relevance, groundedness, and utility.
    """

    def __init__(
        self,
        generator: Any,  # GeneratorModel instance
        rel_tokens: Dict[str, int],
        grd_tokens: Optional[Dict[str, int]] = None,
        ut_tokens: Optional[Dict[str, int]] = None,
        use_groundedness: bool = True,
        use_utility: bool = True,
        w_rel: float = 1.0,
        w_sup: float = 1.0,
        w_use: float = 0.5
    ):
        """
        Args:
            generator: Generator model that produces reflection tokens
            rel_tokens: Relevance token IDs
            grd_tokens: Groundedness token IDs
            ut_tokens: Utility token IDs
            use_groundedness: Use groundedness scores
            use_utility: Use utility scores
            w_rel: Weight for relevance
            w_sup: Weight for groundedness
            w_use: Weight for utility
        """
        self.generator = generator
        self.rel_tokens = rel_tokens
        self.grd_tokens = grd_tokens
        self.ut_tokens = ut_tokens
        self.use_groundedness = use_groundedness
        self.use_utility = use_utility
        self.w_rel = w_rel
        self.w_sup = w_sup
        self.w_use = w_use

    def rerank(
        self,
        query: str,
        documents: List[Document],
        top_k: Optional[int] = None,
        **kwargs
    ) -> RerankResult:
        """
        Rerank documents using LLM reflection token scores.

        Args:
            query: Query string
            documents: Documents to rerank
            top_k: Number of documents to return (None = all)
            **kwargs: Additional arguments

        Returns:
            RerankResult with scored and reranked documents
        """
        if not documents:
            return RerankResult(documents=[], scores=[], metadata={})

        # Generate with each document and extract reflection token scores
        prompts = [self._format_prompt(query, doc) for doc in documents]
        results = self.generator.generate_with_logprobs(prompts)

        # Score each document
        doc_scores = []
        for result in results:
            score = self._compute_score(result)
            doc_scores.append(score)

        # Sort by score (descending)
        sorted_indices = np.argsort(doc_scores)[::-1]
        sorted_docs = [documents[i] for i in sorted_indices]
        sorted_scores = [doc_scores[i] for i in sorted_indices]

        # Limit to top_k
        if top_k is not None:
            sorted_docs = sorted_docs[:top_k]
            sorted_scores = sorted_scores[:top_k]

        return RerankResult(
            documents=sorted_docs,
            scores=sorted_scores,
            metadata={
                "reranker": "LLMCriticReranker",
                "num_scored": len(doc_scores)
            }
        )

    def _format_prompt(self, query: str, document: Document) -> str:
        """Format query + document for critic scoring"""
        title = document.metadata.get("title", "")
        content = document.page_content
        paragraph = f"{title}\n{content}" if title else content
        return f"{query}[Retrieval]<paragraph>{paragraph}</paragraph>\n\n### Response:\n"

    def _compute_score(self, result: Dict[str, Any]) -> float:
        """Extract reflection token scores from generation result"""
        pred_token_ids = result["token_ids"]
        pred_log_probs = result["logprobs"]

        # Extract individual scores
        rel_score = self._extract_relevance_score(pred_log_probs)
        grd_score = (
            self._extract_groundedness_score(pred_token_ids, pred_log_probs)
            if self.use_groundedness and self.grd_tokens
            else 0.0
        )
        ut_score = (
            self._extract_utility_score(pred_token_ids, pred_log_probs)
            if self.use_utility and self.ut_tokens
            else 0.0
        )

        # Weighted sum
        final_score = (
            self.w_rel * rel_score +
            self.w_sup * grd_score +
            self.w_use * ut_score
        )

        return final_score

    def _extract_relevance_score(self, pred_log_probs: List[Dict]) -> float:
        """Extract [Relevant] vs [Irrelevant] score from first token position"""
        if not pred_log_probs:
            return 0.0

        first_token_probs = pred_log_probs[0]
        score_dict = {}

        for tok, token_id in self.rel_tokens.items():
            if token_id in first_token_probs:
                logprob_obj = first_token_probs[token_id]
                if hasattr(logprob_obj, 'logprob'):
                    score_dict[tok] = np.exp(float(logprob_obj.logprob))
                elif isinstance(logprob_obj, (int, float)):
                    score_dict[tok] = np.exp(float(logprob_obj))
                else:
                    score_dict[tok] = 0.0
            else:
                score_dict[tok] = 0.0

        total = sum(score_dict.values())
        return score_dict.get("[Relevant]", 0) / total if total > 0 else 0.0

    def _extract_groundedness_score(self, pred_token_ids: List[int], pred_log_probs: List[Dict]) -> float:
        """Extract [Fully supported]/[Partially supported] score"""
        score_dict = self._find_and_extract_token_probs(pred_token_ids, pred_log_probs, self.grd_tokens)

        if len(score_dict) != 3:
            return 0.0

        total = sum(score_dict.values())
        if total == 0:
            return 0.0

        return (
            score_dict.get("[Fully supported]", 0) / total +
            0.5 * score_dict.get("[Partially supported]", 0) / total
        )

    def _extract_utility_score(self, pred_token_ids: List[int], pred_log_probs: List[Dict]) -> float:
        """Extract [Utility:1-5] score"""
        score_dict = self._find_and_extract_token_probs(pred_token_ids, pred_log_probs, self.ut_tokens)

        if len(score_dict) != 5:
            return 0.0

        total = sum(score_dict.values())
        if total == 0:
            return 0.0

        # Map to [-1, 1] scale
        weights = [-1, -0.5, 0, 0.5, 1]
        return sum(
            weights[i] * score_dict.get(f"[Utility:{i+1}]", 0) / total
            for i in range(5)
        )

    def _find_and_extract_token_probs(
        self,
        pred_token_ids: List[int],
        pred_log_probs: List[Dict],
        token_dict: Dict[str, int]
    ) -> Dict[str, float]:
        """Find where special tokens appear and extract probabilities"""
        for tok_idx, token_id in enumerate(pred_token_ids):
            if token_id in token_dict.values() and tok_idx < len(pred_log_probs):
                token_probs = pred_log_probs[tok_idx]
                result = {}

                for tok, tok_id in token_dict.items():
                    if tok_id in token_probs:
                        logprob_obj = token_probs[tok_id]
                        if hasattr(logprob_obj, 'logprob'):
                            result[tok] = np.exp(float(logprob_obj.logprob))
                        elif isinstance(logprob_obj, (int, float)):
                            result[tok] = np.exp(float(logprob_obj))
                        else:
                            result[tok] = 0.0
                    else:
                        result[tok] = 0.0

                return result

        return {}
