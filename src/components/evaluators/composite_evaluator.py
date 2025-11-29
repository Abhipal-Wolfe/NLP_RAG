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
        ground_truths: List[str],
        **kwargs
    ) -> EvaluationResult:
        """
        Run all evaluators and combine metrics.

        Args:
            predictions: Model predictions
            ground_truths: Ground truth answers
            **kwargs: Additional arguments

        Returns:
            EvaluationResult with combined metrics
        """
        combined_metrics = {}

        for evaluator in self.evaluators:
            result = evaluator.evaluate(predictions, ground_truths, **kwargs)
            # Merge metrics with namespace to avoid conflicts
            evaluator_name = evaluator.__class__.__name__
            for key, value in result.metrics.items():
                combined_key = f"{evaluator_name}.{key}" if len(self.evaluators) > 1 else key
                combined_metrics[combined_key] = value

        return EvaluationResult(
            metrics=combined_metrics,
            predictions=predictions,
            ground_truths=ground_truths,
            metadata={
                "evaluator": "CompositeEvaluator",
                "num_evaluators": len(self.evaluators)
            }
        )
