"""Composite evaluator that combines multiple metrics"""

from typing import List, Dict, Any
from ...core.interfaces import Evaluator, EvaluationResult
from ...core.registry import register_component


@register_component("evaluator", "CompositeEvaluator")
class CompositeEvaluator(Evaluator):
    """
    Composite evaluator that runs multiple evaluators and combines results.

    Useful for computing multiple metrics (accuracy, ROUGE, BERTScore, etc.) in one pass.
    """

    def __init__(self, evaluators: List[Evaluator]):
        """
        Args:
            evaluators: List of evaluator instances
        """
        self.evaluators = evaluators

    def evaluate(
        self,
        predictions: List[str],
        ground_truths: List[Any],
        **kwargs
    ) -> EvaluationResult:
        """
        Run all evaluators and combine metrics.

        Args:
            predictions: Model predictions (strings)
            ground_truths: Ground truth answers; can be strings or
                structured objects (e.g., dicts with answer_idx/answer_text
                for MCQ tasks like MEDQA).
            **kwargs: Additional arguments passed through to each evaluator.

        Returns:
            EvaluationResult with combined metrics.
        """
        combined_metrics: Dict[str, Any] = {}

        for evaluator in self.evaluators:
            result = evaluator.evaluate(predictions, ground_truths, **kwargs)

            # Merge metrics with namespace to avoid conflicts when >1 evaluator
            evaluator_name = evaluator.__class__.__name__
            for key, value in result.metrics.items():
                if len(self.evaluators) > 1:
                    combined_key = f"{evaluator_name}.{key}"
                else:
                    combined_key = key
                combined_metrics[combined_key] = value

        return EvaluationResult(
            metrics=combined_metrics,
            predictions=predictions,
            ground_truths=ground_truths,
            metadata={
                "evaluator": "CompositeEvaluator",
                "num_evaluators": len(self.evaluators),
            },
        )