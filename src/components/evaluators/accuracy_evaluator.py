"""Accuracy evaluator for QA tasks"""

from typing import List, Dict, Any
import re
import string

from ...core.interfaces import Evaluator, EvaluationResult
from ...core.registry import register_component


@register_component("evaluator", "AccuracyEvaluator")
class AccuracyEvaluator(Evaluator):
    """
    Evaluator that computes accuracy metrics.

    Supports:
    - Substring match: Check if ground truth appears in prediction
    - Exact match: Normalized exact match after removing articles/punctuation
    """

    def __init__(self, use_substring_match: bool = True, use_exact_match: bool = True):
        """
        Args:
            use_substring_match: Compute substring match accuracy
            use_exact_match: Compute exact match accuracy
        """
        self.use_substring_match = use_substring_match
        self.use_exact_match = use_exact_match

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
            **kwargs: Additional arguments

        Returns:
            EvaluationResult with accuracy metrics
        """
        if len(predictions) != len(ground_truths):
            raise ValueError(
                f"Predictions ({len(predictions)}) and ground truths ({len(ground_truths)}) "
                "must have the same length"
            )

        metrics = {}

        # Substring match
        if self.use_substring_match:
            substring_correct = sum(
                1 for pred, gt in zip(predictions, ground_truths)
                if gt.lower() in pred.lower()
            )
            metrics["substring_match"] = {
                "accuracy": 100 * substring_correct / len(predictions) if predictions else 0,
                "correct": substring_correct,
                "total": len(predictions)
            }

        # Exact match
        if self.use_exact_match:
            exact_correct = sum(
                1 for pred, gt in zip(predictions, ground_truths)
                if self._normalize_answer(pred) == self._normalize_answer(gt)
            )
            metrics["exact_match"] = {
                "accuracy": 100 * exact_correct / len(predictions) if predictions else 0,
                "correct": exact_correct,
                "total": len(predictions)
            }

        # Determine primary metric
        if self.use_substring_match:
            metrics["primary_method"] = "substring_match"
        elif self.use_exact_match:
            metrics["primary_method"] = "exact_match"

        return EvaluationResult(
            metrics=metrics,
            predictions=predictions,
            ground_truths=ground_truths,
            metadata={
                "evaluator": "AccuracyEvaluator",
                "use_substring_match": self.use_substring_match,
                "use_exact_match": self.use_exact_match
            }
        )

    def _normalize_answer(self, s: str) -> str:
        """Normalize answer for exact match evaluation"""
        def remove_articles(text):
            return re.sub(r'\b(a|an|the)\b', ' ', text)

        def white_space_fix(text):
            return ' '.join(text.split())

        def remove_punc(text):
            exclude = set(string.punctuation)
            return ''.join(ch for ch in text if ch not in exclude)

        def lower(text):
            return text.lower()

        return white_space_fix(remove_articles(remove_punc(lower(s))))
