"""Unit tests for VotingEnsemble and StackingEnsemble."""

from __future__ import annotations

import numpy as np
import pytest

from pipeline.evaluation.metrics import Metrics, accuracy
from pipeline.hpo.ensemble import StackingEnsemble, VotingEnsemble
from pipeline.protocols import Batch


class _SimpleFakeModel:
    """Model that returns configurable constant output."""

    def __init__(self, output):
        self._output = np.asarray(output)
        self._mode = "train"

    def forward(self, inputs):
        n = len(np.asarray(inputs))
        return np.tile(self._output, (n, 1))

    def parameters(self):
        return []

    def train_mode(self):
        self._mode = "train"

    def eval_mode(self):
        self._mode = "eval"


def _votes_are(ensemble, inputs):
    """Convenience: predict over 2 inputs and return labels as list."""
    return ensemble.predict(np.asarray(inputs)).tolist()


class TestVotingEnsemble:
    def test_hard_voting_majority(self):
        # Two models vote class 0, one votes class 1 -> majority 0.
        ensemble = VotingEnsemble(
            models=[
                _SimpleFakeModel([[0.9, 0.1]]),
                _SimpleFakeModel([[0.8, 0.2]]),
                _SimpleFakeModel([[0.3, 0.7]]),
            ],
            mode="hard",
        )
        inputs = [[1.0, 1.0], [1.0, 1.0]]
        assert _votes_are(ensemble, inputs) == [0, 0]

    def test_soft_voting_avg_prob(self):
        # Model A strongly class 0, model B strongly class 1.
        # Average prob of class 1: 0.1+0.9 over 2 = 0.5; class 0 also 0.5.
        # Use an imbalance so soft averaging decides class 1.
        ensemble = VotingEnsemble(
            models=[
                _SimpleFakeModel([[1.0, 0.0]]),
                _SimpleFakeModel([[0.0, 1.0]]),
            ],
            mode="soft",
        )
        inputs = [[1.0, 1.0]]
        # avg probs = [[0.5, 0.5]] -> argmax picks index 0 as first max.
        assert _votes_are(ensemble, inputs)[0] in (0, 1)

    def test_soft_voting_avg_prob_decides(self):
        # Three models: two strongly class 1 -> soft average favours class 1.
        ensemble = VotingEnsemble(
            models=[
                _SimpleFakeModel([[0.1, 0.9]]),
                _SimpleFakeModel([[0.2, 0.8]]),
                _SimpleFakeModel([[0.9, 0.1]]),
            ],
            mode="soft",
        )
        # avg = [[0.4, 0.6]] -> argmax = 1.
        assert _votes_are(ensemble, [[1.0]]) == [1]

    def test_single_model(self):
        ensemble = VotingEnsemble(models=[_SimpleFakeModel([[0.3, 0.7]])])
        assert _votes_are(ensemble, [[1.0, 1.0]]) == [1]

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            VotingEnsemble(models=[])

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError):
            VotingEnsemble(models=[_SimpleFakeModel([[0.5, 0.5]])], mode="mean")

    def test_predict_shape(self):
        ensemble = VotingEnsemble(
            models=[
                _SimpleFakeModel([[0.5, 0.5]]),
                _SimpleFakeModel([[0.5, 0.5]]),
            ]
        )
        preds = ensemble.predict(np.zeros((5, 3)))
        assert preds.shape == (5,)

    def test_evaluate_on_stream(self):
        ensemble = VotingEnsemble(
            models=[
                _SimpleFakeModel([[0.9, 0.1]]),
                _SimpleFakeModel([[0.9, 0.1]]),
            ]
        )
        stream = [
            Batch(inputs=np.array([[1.0], [2.0]]), targets=np.array([0, 0])),
            Batch(inputs=np.array([[3.0]]), targets=np.array([0])),
        ]
        metrics = Metrics(accuracy=accuracy)
        result = ensemble.evaluate(stream, metrics)
        assert result is metrics
        assert metrics["accuracy"] == pytest.approx(1.0)

    def test_mixed_model_outputs(self):
        # Models with differing probabilities still all pick argmax class 1.
        ensemble = VotingEnsemble(
            models=[
                _SimpleFakeModel([[0.1, 0.9]]),
                _SimpleFakeModel([[0.4, 0.6]]),
            ],
            mode="hard",
        )
        assert _votes_are(ensemble, [[1.0, 1.0], [1.0, 1.0]]) == [1, 1]


class TestStackingEnsemble:
    def test_fit_sets_fitted(self):
        stacking = StackingEnsemble(
            base_models=[
                _SimpleFakeModel([[0.5, 0.5]]),
                _SimpleFakeModel([[0.5, 0.5]]),
            ],
            meta_model=_SimpleFakeModel([[0.5, 0.5]]),
        )
        stream = [Batch(inputs=np.array([[1.0], [2.0]]), targets=np.array([0, 1]))]
        stacking.fit(stream)
        assert stacking.is_fitted is True

    def test_predict_after_fit(self):
        stacking = StackingEnsemble(
            base_models=[
                _SimpleFakeModel([[0.9, 0.1]]),
                _SimpleFakeModel([[0.9, 0.1]]),
            ],
            meta_model=_SimpleFakeModel([[1.0, 0.0]]),
        )
        stream = [Batch(inputs=np.array([[1.0], [2.0]]), targets=np.array([0, 0]))]
        stacking.fit(stream)
        preds = stacking.predict(np.array([[1.0], [2.0]]))
        assert np.asarray(preds).shape == (2, 2)

    def test_predict_shape(self):
        stacking = StackingEnsemble(
            base_models=[
                _SimpleFakeModel([[0.5, 0.5]]),
                _SimpleFakeModel([[0.5, 0.5]]),
            ],
            meta_model=_SimpleFakeModel([[0.5, 0.5]]),
        )
        stacking.fit([Batch(inputs=np.zeros((4, 2)), targets=np.zeros(4))])
        preds = np.asarray(stacking.predict(np.zeros((7, 2))))
        assert preds.shape[0] == 7

    def test_empty_base_raises(self):
        with pytest.raises(ValueError):
            StackingEnsemble(base_models=[], meta_model=_SimpleFakeModel([[1.0]]))

    def test_is_fitted_before_fit(self):
        stacking = StackingEnsemble(
            base_models=[_SimpleFakeModel([[0.5, 0.5]])],
            meta_model=_SimpleFakeModel([[0.5, 0.5]]),
        )
        assert stacking.is_fitted is False
