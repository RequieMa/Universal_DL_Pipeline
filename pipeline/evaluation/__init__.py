"""Evaluation metrics for pipeline outputs.

Provides pure numpy-based metric functions and a :class:`Metrics`
container that decouples the evaluate stage from specific metric choices.
"""

from pipeline.evaluation.metrics import Metrics, accuracy, confusion_matrix, f1_score, precision, recall

__all__ = ["Metrics", "accuracy", "confusion_matrix", "f1_score", "precision", "recall"]
