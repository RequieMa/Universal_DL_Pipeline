"""Ensemble methods — voting and stacking over trained models."""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING

import numpy as np

from pipeline.protocols import ArrayLike, ModelProtocol

if TYPE_CHECKING:
    from pipeline.evaluation.metrics import Metrics
    from pipeline.protocols import DataStream


class VotingEnsemble:
    """Combine predictions by majority vote (classification) or mean (regression).

    All base models must satisfy :class:`ModelProtocol`. Models are assumed
    already trained.

    Usage::

        ensemble = VotingEnsemble(models=[m1, m2, m3], mode="hard")
        preds = ensemble.predict(test_inputs)

    Args:
        models: List of trained :class:`ModelProtocol` instances.
        mode: ``"hard"`` — majority vote on class labels.
            ``"soft"`` — average class probabilities, then argmax.
    """

    def __init__(self, models: list[ModelProtocol], mode: str = "hard") -> None:
        if not models:
            raise ValueError("models must not be empty")
        if mode not in ("hard", "soft"):
            raise ValueError(f"mode must be 'hard' or 'soft', got {mode!r}")
        self._models = models
        self._mode = mode

    def predict(self, inputs: ArrayLike) -> np.ndarray:
        """Run all models and aggregate predictions.

        Args:
            inputs: Input features.

        Returns:
            Aggregated class labels, shape ``(n_samples,)``.
        """
        all_preds = [np.asarray(m.forward(inputs)) for m in self._models]

        if self._mode == "hard":
            # Each model returns class probabilities or labels
            # Take argmax per model, then majority vote
            votes = np.stack([np.argmax(p, axis=-1) for p in all_preds], axis=0)
            result = []
            for i in range(votes.shape[1]):
                counts = Counter(votes[:, i].tolist())
                result.append(counts.most_common(1)[0][0])
            return np.array(result)

        # Soft voting: average probabilities, then argmax
        stacked = np.stack(all_preds, axis=0)  # (n_models, n_samples, n_classes)
        avg_probs = np.mean(stacked, axis=0)
        return np.argmax(avg_probs, axis=-1)

    def evaluate(self, data_stream: DataStream, metrics: Metrics) -> Metrics:
        """Evaluate ensemble on a data stream.

        Args:
            data_stream: Stream yielding :class:`Batch` objects.
            metrics: :class:`Metrics` instance with desired metric functions.

        Returns:
            The same :class:`Metrics` instance with computed values.
        """
        all_preds = []
        all_targets = []
        for batch in data_stream:
            preds = self.predict(batch.inputs)
            all_preds.append(preds)
            all_targets.append(np.asarray(batch.targets))
        y_pred = np.concatenate(all_preds)
        y_true = np.concatenate(all_targets)
        metrics.compute(y_true, y_pred)
        return metrics


class StackingEnsemble:
    """Train a meta-model on base model predictions.

    Base models produce predictions on training data. A meta-model
    learns to combine them. Both base and meta models satisfy
    :class:`ModelProtocol`.

    Usage::

        stacking = StackingEnsemble(
            base_models=[m1, m2],
            meta_model=SklearnModel(LogisticRegression()),
        )
        stacking.fit(train_stream)
        preds = stacking.predict(test_inputs)

    Args:
        base_models: List of trained :class:`ModelProtocol` instances.
        meta_model: A :class:`ModelProtocol` instance to combine outputs.
    """

    def __init__(
        self, base_models: list[ModelProtocol], meta_model: ModelProtocol
    ) -> None:
        if not base_models:
            raise ValueError("base_models must not be empty")
        self._base_models = base_models
        self._meta_model = meta_model
        self._fitted = False

    def fit(self, data_stream: DataStream) -> StackingEnsemble:
        """Train meta-model on stacked base model outputs.

        Iterates ``data_stream``, runs each base model, stacks their
        outputs as features, and fits the meta-model.

        Args:
            data_stream: Training data stream.

        Returns:
            ``self`` for method chaining.
        """
        all_features = []
        all_targets = []

        for batch in data_stream:
            base_outputs = [
                np.asarray(m.forward(batch.inputs)) for m in self._base_models
            ]
            # Stack base model outputs as features: (batch, n_models * n_classes)
            features = np.concatenate(base_outputs, axis=-1)
            all_features.append(features)
            all_targets.append(np.asarray(batch.targets))

        X = np.concatenate(all_features, axis=0)
        y = np.concatenate(all_targets)

        # Train meta-model using its own fit if available (sklearn),
        # otherwise just call forward to validate
        if hasattr(self._meta_model, "_estimator") and hasattr(
            self._meta_model._estimator, "fit"
        ):
            self._meta_model._estimator.fit(X, y)

        self._fitted = True
        return self

    def predict(self, inputs: ArrayLike) -> np.ndarray:
        """Predict using base models + trained meta-model.

        Args:
            inputs: Input features.

        Returns:
            Final predictions after meta-model combination.
        """
        base_outputs = [
            np.asarray(m.forward(inputs)) for m in self._base_models
        ]
        features = np.concatenate(base_outputs, axis=-1)
        return np.asarray(self._meta_model.forward(features))

    @property
    def is_fitted(self) -> bool:
        """Whether :meth:`fit` has been called."""
        return self._fitted
