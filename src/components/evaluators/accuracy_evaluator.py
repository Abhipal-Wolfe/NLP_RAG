"""Accuracy evaluator for QA tasks"""

from typing import List, Dict, Any, Tuple
import re
import string

from ...core.interfaces import Evaluator, EvaluationResult
from ...core.registry import register_component


@register_component("evaluator", "AccuracyEvaluator")
class AccuracyEvaluator(Evaluator):
    """
    Evaluator that computes accuracy metrics for multiple-choice QA.
    """

    def __init__(self):
        """
        Computes accuracy based on multiple-choice option letters (A/B/C/D),
        and, when possible, also checks that the chosen option text matches
        the ground-truth answer_text.
        """
        pass

    def evaluate(
        self,
        predictions: List[str],
        ground_truths: List[Any],
        **kwargs
    ) -> EvaluationResult:
        """
        Evaluate predictions against ground truths using MCQ choice-letter accuracy.

        An answer is counted as correct if:
          1. The predicted choice letter equals the ground-truth letter, AND
          2. If a ground-truth answer_text is available:
              the normalized predicted option text matches the normalized
              ground-truth answer_text.

        Args:
            predictions: Model predictions (strings)
            ground_truths: Ground truth items (dicts with answer_idx / answer_text,
                           or strings)
            **kwargs: Additional arguments (unused)

        Returns:
            EvaluationResult with 'choice_match' accuracy
        """
        if len(predictions) != len(ground_truths):
            raise ValueError(
                f"Predictions ({len(predictions)}) and ground truths ({len(ground_truths)}) "
                "must have the same length"
            )

        metrics: Dict[str, Any] = {}

        choice_correct = 0
        choice_total = 0

        for pred, gt in zip(predictions, ground_truths):
            pred_letter, pred_text = self._parse_prediction(pred)
            gt_letter, gt_text = self._parse_ground_truth(gt)

            # Only evaluate if we have a valid ground-truth letter
            if not gt_letter:
                continue

            choice_total += 1
            is_correct = False

            if pred_letter == gt_letter:
                if gt_text:
                    # We have canonical ground-truth text; require predicted text
                    # to match it (after normalization)
                    if pred_text:
                        if self._normalize_text(pred_text) == self._normalize_text(gt_text):
                            is_correct = True
                        else:
                            is_correct = False
                    else:
                        # No parsable predicted text; be conservative and mark incorrect
                        is_correct = False
                else:
                    # No ground-truth text available; letter match is enough
                    is_correct = True

            if is_correct:
                choice_correct += 1

        metrics["choice_match"] = {
            "accuracy": 100 * choice_correct / choice_total if choice_total > 0 else 0.0,
            "correct": choice_correct,
            "total": choice_total,
        }
        metrics["primary_method"] = "choice_match"

        return EvaluationResult(
            metrics=metrics,
            predictions=predictions,
            ground_truths=ground_truths,
            metadata={
                "evaluator": "AccuracyEvaluator",
                "mode": "choice_match",
            },
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _parse_prediction(self, text: str) -> Tuple[str, str]:
        """
        Parse model prediction into (choice_letter, option_text).

        - Prefer to look after 'Answer:' (case-insensitive).
        - First try patterns like '<LETTER>. <text>'.
        - If that fails, fall back to finding a standalone LETTER only.

        Returns:
            (letter, text) where letter is 'A'...'D' or '' if not found,
            and text is the option text (possibly empty).
        """
        if not text:
            return "", ""

        t = text.strip()
        if not t:
            return "", ""

        # Locate 'Answer:' marker (case-insensitive)
        lower_t = t.lower()
        marker = "answer:"
        start_idx = lower_t.find(marker)
        if start_idx != -1:
            region = t[start_idx + len(marker):].strip()
        else:
            region = t

        # 1) Try pattern: 'A. some text' or '(A) some text'
        m = re.search(r"[ \t]*\(?([A-D])\)?[.)]\s*(.+)", region)
        if m:
            letter = m.group(1)
            opt_text = m.group(2).strip()
            return letter, opt_text

        # 2) Try standalone letter (no text reliably parsed)
        m2 = re.search(r"\b([A-D])\b", region)
        if m2:
            letter = m2.group(1)
            return letter, ""

        return "", ""

    def _parse_ground_truth(self, gt: Any) -> Tuple[str, str]:
        """
        Parse ground truth into (choice_letter, option_text).

        Dict case (MEDQA-style):
            letter  <- answer_idx
            text    <- answer_text
        String case:
            try to parse formats like 'B. some text' or '(C) some text',
            or at least extract a standalone 'C'.
        """
        # Dict case
        if isinstance(gt, dict):
            letter = ""
            text = ""

            # 1) answer_idx
            idx = gt.get("answer_idx")
            if idx is not None:
                idx_str = str(idx).strip().upper()
                if len(idx_str) == 1 and idx_str in "ABCD":
                    letter = idx_str

            # 2) answer_text
            answer_text = gt.get("answer_text")
            if isinstance(answer_text, str) and answer_text.strip():
                text = answer_text.strip()

            return letter, text

        # String case
        if isinstance(gt, str):
            t = gt.strip()
            if not t:
                return "", ""

            # Try pattern like '(C) something' or 'C. something'
            m = re.match(r"^\s*\(?([A-D])\)?[.)\s]*(.*)$", t)
            if m:
                letter = m.group(1)
                opt_text = m.group(2).strip()
                return letter, opt_text

            # Try standalone letter anywhere
            m2 = re.search(r"\b([A-D])\b", t)
            if m2:
                return m2.group(1), ""

        return "", ""

    def _normalize_text(self, s: str) -> str:
        """
        Normalize option text for comparison:
        - Lowercase
        - Remove punctuation
        - Collapse multiple spaces
        """
        if not s:
            return ""
        # Remove punctuation
        translator = str.maketrans("", "", string.punctuation)
        s = s.translate(translator)
        # Lowercase and normalize spaces
        s = " ".join(s.lower().split())
        return s