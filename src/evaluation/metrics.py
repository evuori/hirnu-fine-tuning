"""Custom metrics for Hirnu model evaluation."""

from typing import List, Dict, Any
import re


class HirnuMetrics:
    """Metrics for evaluating Hirnu language model."""

    @staticmethod
    def calculate_perplexity(loss: float) -> float:
        """Calculate perplexity from loss.

        Args:
            loss: Cross-entropy loss

        Returns:
            Perplexity value
        """
        import math

        return math.exp(loss)

    @staticmethod
    def bleu_score(
        predictions: List[str], references: List[str], max_n: int = 4
    ) -> float:
        """Calculate BLEU score for translation quality.

        Args:
            predictions: List of predicted translations
            references: List of reference translations
            max_n: Maximum n-gram order

        Returns:
            BLEU score (0-100)
        """
        # TODO: Implement BLEU score calculation
        # For production, consider using sacrebleu library
        # This is a simplified placeholder

        if len(predictions) != len(references):
            raise ValueError("Predictions and references must have same length")

        # Placeholder implementation
        score = 0.0

        return score

    @staticmethod
    def character_error_rate(predictions: List[str], references: List[str]) -> float:
        """Calculate Character Error Rate (CER).

        Args:
            predictions: List of predicted texts
            references: List of reference texts

        Returns:
            Character error rate (0-1)
        """
        if len(predictions) != len(references):
            raise ValueError("Predictions and references must have same length")

        total_chars = 0
        total_errors = 0

        for pred, ref in zip(predictions, references):
            total_chars += len(ref)
            # Simple character-level edit distance
            errors = sum(1 for p, r in zip(pred, ref) if p != r)
            errors += abs(len(pred) - len(ref))
            total_errors += errors

        return total_errors / total_chars if total_chars > 0 else 0.0

    @staticmethod
    def exact_match_accuracy(predictions: List[str], references: List[str]) -> float:
        """Calculate exact match accuracy.

        Args:
            predictions: List of predicted texts
            references: List of reference texts

        Returns:
            Exact match accuracy (0-1)
        """
        if len(predictions) != len(references):
            raise ValueError("Predictions and references must have same length")

        matches = sum(1 for pred, ref in zip(predictions, references) if pred == ref)
        return matches / len(predictions) if predictions else 0.0

    @staticmethod
    def word_accuracy(predictions: List[str], references: List[str]) -> float:
        """Calculate word-level accuracy.

        Args:
            predictions: List of predicted texts
            references: List of reference texts

        Returns:
            Word accuracy (0-1)
        """
        if len(predictions) != len(references):
            raise ValueError("Predictions and references must have same length")

        total_words = 0
        correct_words = 0

        for pred, ref in zip(predictions, references):
            pred_words = pred.split()
            ref_words = ref.split()

            total_words += len(ref_words)
            correct_words += sum(1 for p, r in zip(pred_words, ref_words) if p == r)

        return correct_words / total_words if total_words > 0 else 0.0


class TranslationMetrics:
    """Metrics specific to English-Hirnu translation."""

    @staticmethod
    def evaluate_translation(
        english_input: str,
        hirnu_prediction: str,
        hirnu_reference: str,
    ) -> Dict[str, float]:
        """Evaluate a single translation.

        Args:
            english_input: English input text
            hirnu_prediction: Predicted Hirnu translation
            hirnu_reference: Reference Hirnu translation

        Returns:
            Dictionary of metrics
        """
        metrics = {}

        # Exact match
        metrics["exact_match"] = (
            1.0 if hirnu_prediction == hirnu_reference else 0.0
        )

        # Character error rate
        cer = HirnuMetrics.character_error_rate(
            [hirnu_prediction], [hirnu_reference]
        )
        metrics["character_error_rate"] = cer

        # Word accuracy
        word_acc = HirnuMetrics.word_accuracy(
            [hirnu_prediction], [hirnu_reference]
        )
        metrics["word_accuracy"] = word_acc

        return metrics

    @staticmethod
    def evaluate_batch(
        english_inputs: List[str],
        hirnu_predictions: List[str],
        hirnu_references: List[str],
    ) -> Dict[str, float]:
        """Evaluate a batch of translations.

        Args:
            english_inputs: List of English inputs
            hirnu_predictions: List of predicted Hirnu translations
            hirnu_references: List of reference Hirnu translations

        Returns:
            Dictionary of aggregated metrics
        """
        if not (
            len(english_inputs) == len(hirnu_predictions) == len(hirnu_references)
        ):
            raise ValueError("All input lists must have the same length")

        # Calculate metrics for each example
        all_metrics = []
        for eng, pred, ref in zip(
            english_inputs, hirnu_predictions, hirnu_references
        ):
            metrics = TranslationMetrics.evaluate_translation(eng, pred, ref)
            all_metrics.append(metrics)

        # Aggregate metrics
        aggregated = {
            "exact_match_rate": sum(m["exact_match"] for m in all_metrics)
            / len(all_metrics),
            "avg_character_error_rate": sum(
                m["character_error_rate"] for m in all_metrics
            )
            / len(all_metrics),
            "avg_word_accuracy": sum(m["word_accuracy"] for m in all_metrics)
            / len(all_metrics),
        }

        return aggregated
