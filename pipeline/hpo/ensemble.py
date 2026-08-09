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

        Base models are not required to return the same kind of output:
        some return 2D class probabilities ``(n_samples, n_classes)``,
        others return 1D class labels ``(n_samples,)``. Hard voting
        normalizes both to labels before voting; soft voting requires
        probabilities (2D) because it averages them.

        Args:
            inputs: Input features.

        Returns:
            Aggregated class labels, shape ``(n_samples,)``.

        Raises:
            ValueError: In hard mode, if a model output has an
                unsupported number of dimensions or the per-model label
                vectors disagree on the number of samples. In soft mode,
                if any model returns 1D labels instead of probabilities.
        """
        all_preds = [np.asarray(m.forward(inputs)) for m in self._models]

        if self._mode == "hard":
            # Normalize each model's output to a 1D label vector:
            #   1D -> already labels, use as-is
            #   2D -> class probabilities, argmax over the class axis
            #   otherwise -> unsupported
            label_vecs = []
            for p in all_preds:
                if p.ndim == 1:
                    labels = p
                elif p.ndim == 2:
                    labels = np.argmax(p, axis=1)
                else:
                    raise ValueError(
                        "hard voting expects each model output to be 1D "
                        f"labels or 2D probabilities, got {p.ndim}D array"
                    )
                label_vecs.append(np.asarray(labels).ravel())

            n_samples = label_vecs[0].shape[0]
            if any(v.shape[0] != n_samples for v in label_vecs):
                raise ValueError(
                    "all models must predict the same number of samples; "
                    f"got sample counts {[int(v.shape[0]) for v in label_vecs]}"
                )

            # Majority vote across models, per sample.
            votes = np.stack(label_vecs, axis=0)  # (n_models, n_samples)
            result = []
            for i in range(n_samples):
                counts = Counter(votes[:, i].tolist())
                result.append(counts.most_common(1)[0][0])
            return np.array(result)

        # Soft voting: average probabilities, then argmax.
        # Averaging is only meaningful for probability outputs (2D).
        for p in all_preds:
            if p.ndim != 2:
                raise ValueError(
                    "soft voting needs probability outputs of shape "
                    f"(n_samples, n_classes), got a {p.ndim}D array; use "
                    'mode="hard" for models that return class labels'
                )
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

    The meta-model **must be trainable**: it either exposes a public
    ``fit(X, y)`` method, or it is a sklearn-backed adapter with a
    fittable ``._estimator`` (e.g. :class:`SklearnModel`). If neither
    is available, :meth:`fit` raises rather than silently doing nothing.

    Usage::

        stacking = StackingEnsemble(
            base_models=[m1, m2],
            meta_model=SklearnModel(LogisticRegression()),
        )
        stacking.fit(train_stream)
        preds = stacking.predict(test_inputs)

    Args:
        base_models: List of trained :class:`ModelProtocol` instances.
        meta_model: A trainable model to combine base outputs — either
            with a public ``fit(X, y)`` or a sklearn-backed ``._estimator``.
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

        Raises:
            TypeError: If the meta-model is not trainable — it exposes
                neither a public ``fit(X, y)`` method nor a sklearn-backed
                ``._estimator`` with ``fit``. ``is_fitted`` stays ``False``
                in that case (the meta-model is never silently skipped).
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

        # Train the meta-model. Prefer a public duck-typed fit(X, y); fall
        # back to the sklearn adapter's private ._estimator.fit for
        # backward compatibility. If neither exists, fail loudly instead of
        # marking the ensemble fitted with an untrained meta-model.
        meta_fit = getattr(self._meta_model, "fit", None)
        estimator = getattr(self._meta_model, "_estimator", None)
        estimator_fit = getattr(estimator, "fit", None)
        if callable(meta_fit):
            meta_fit(X, y)
        elif callable(estimator_fit):
            estimator_fit(X, y)
        else:
            raise TypeError(
                "StackingEnsemble meta_model must be trainable: expected a "
                "public .fit(X, y) or a sklearn-backed adapter with "
                f"._estimator.fit; got {type(self._meta_model).__name__}"
            )

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
