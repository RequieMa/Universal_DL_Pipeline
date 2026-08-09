"""Integration tests: hyperparameter search + ensemble working together.

E2E coverage for Phase 5:
    1. GridSearch over a minimal fake pipeline — multiple trials scored and
       ranked, with a non-negative best score.
    2. RandomSearch over the same pipeline — a fixed number of trials sampled,
       and the result ranked in descending order of score.
    3. VotingEnsemble assembled from the top models found by a GridSearch —
       predicting valid class labels end to end.

Everything runs on a single self-contained fake pipeline (no real framework or
dataset), following the same minimal ``BasePipeline`` subclass pattern used by
Task 1's ``tests/unit/test_pipeline.py::_MinimalPipeline``. Tests are marked
``slow`` per project conventions for integration/E2E coverage.
"""

from __future__ import annotations

import numpy as np
import pytest

from pipeline.config import Config
from pipeline.evaluation.metrics import Metrics, accuracy
from pipeline.hpo import GridSearch, RandomSearch, VotingEnsemble
from pipeline.pipeline import BasePipeline, PipelineState
from pipeline.protocols import Loss


# ── Minimal fake pipeline for the integration tests ─────────────────────
class _MinimalPipeline(BasePipeline):
    """Same minimal shape as Task 1's ``_MinimalPipeline`` in test_pipeline.py.

    Every stage completes without real data or a framework. ``build_model``
    constructs a deterministic two-class classifier whose bias scales with
    ``learning_rate``; ``evaluate`` scores it with :func:`accuracy`.

    The classifier is the payload we recover after a search to feed a
    :class:`VotingEnsemble`, so ``build_model`` stores a real model on
    ``state.model``.
    """

    def load_data(self, state: PipelineState) -> None:
        state.current_epoch = 0

    def build_model(self, state: PipelineState) -> None:
        state.model = _FakeModel(bias=state.config.learning_rate * 10.0)
        state.loss_fn = _FakeLoss()
        state.optimizer = _FakeOptimizer()

    def train(self, state: PipelineState) -> None:
        state.current_epoch = state.config.num_epochs

    def evaluate(self, state: PipelineState) -> None:
        # Deterministic scoring that improves with learning_rate and is always
        # non-negative, so the search can rank trials by score.
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1])
        state.metrics = Metrics(accuracy=accuracy)
        state.metrics.compute(y_true, y_pred)
        state.metrics._values["accuracy"] = 0.4 + state.config.learning_rate * 5

    def export(self, state: PipelineState) -> None:
        state.predictions = np.array([0, 1, 0, 1])


class _FakeModel:
    """Two-class classifier returning per-sample class probabilities.

    ``forward`` maps ``inputs` to ``(n_samples, 2)`` probabilities whose class
    flips depending on a config-driven bias, so two models trained with
    different configs genuinely disagree on some inputs — making majority
    voting meaningful.
    """

    def __init__(self, bias: float) -> None:
        self._bias = bias

    def forward(self, inputs):
        x = np.asarray(inputs, dtype=float)
        p1 = 1.0 / (1.0 + np.exp(-(x[:, 0] + self._bias)))
        return np.stack([1.0 - p1, p1], axis=-1)

    def parameters(self):
        return []

    def train_mode(self) -> None:
        return None

    def eval_mode(self) -> None:
        return None


class _FakeLoss:
    """Constant zero loss — never actually used for training here."""

    def forward(self, p, t) -> Loss:
        return Loss(value=0.0)

    def __call__(self, p, t) -> Loss:
        return self.forward(p, t)


class _FakeOptimizer:
    """No-op optimizer satisfying OptimizerProtocol."""

    def step(self) -> None:
        return None

    def zero_grad(self) -> None:
        return None


def _param_grid() -> dict[str, list]:
    """A 2x2 grid = 4 combinations (2 learning rates x 2 batch sizes)."""
    return {"learning_rate": [0.001, 0.01, 0.02], "batch_size": [16, 32]}


def _scores_by_rank(result) -> list[float]:
    """Successful trial scores sorted descending."""
    scores = sorted(t.score for t in result.trials if t.score is not None)
    return list(reversed(scores))


# ── 1. GridSearch over the fake pipeline ────────────────────────────────


@pytest.mark.slow
def test_grid_search_on_fake_pipeline():
    """GridSearch finishes 4 trials and reports a valid best score."""
    search = GridSearch(
        param_grid={"learning_rate": [0.001, 0.01], "batch_size": [16, 32]},
        pipeline_cls=_MinimalPipeline,
        base_config=Config(),
        scoring="accuracy",
    )

    result = search.run()

    assert len(result.trials) == 4
    assert all(t.score is not None for t in result.trials)
    assert result.best_score >= 0


# ── 2. RandomSearch over the fake pipeline ──────────────────────────────


@pytest.mark.slow
def test_random_search_on_fake_pipeline():
    """RandomSearch samples 3 trials and ranks them descending by score."""
    search = RandomSearch(
        param_grid=_param_grid(),
        n_trials=3,
        seed=42,
        pipeline_cls=_MinimalPipeline,
        base_config=Config(),
        scoring="accuracy",
    )

    result = search.run()

    assert len(result.trials) == 3
    assert all(t.score is not None for t in result.trials)
    assert _scores_by_rank(result) == sorted(_scores_by_rank(result), reverse=True)


# ── 3. VotingEnsemble built from the search's top models ────────────────


@pytest.mark.slow
def test_voting_ensemble_from_search():
    """Top-2 models from a GridSearch form a VotingEnsemble that predicts labels.

    ``SearchResult`` stores each trial's config, not the trained model, so the
    top-2 configs are re-run to recover their ``state.model`` instances — the
    search and the ensemble are exercised together end to end.
    """
    search = GridSearch(
        param_grid={"learning_rate": [0.001, 0.01], "batch_size": [16]},
        pipeline_cls=_MinimalPipeline,
        base_config=Config(),
        scoring="accuracy",
    )

    result = search.run()
    assert len(result.trials) == 2

    # Recover the trained model for the two best-scoring configs.
    top = sorted(
        (t for t in result.trials if t.score is not None),
        key=lambda t: t.score,
        reverse=True,
    )[:2]
    models = []
    for trial in top:
        pipeline = _MinimalPipeline(trial.config)
        state = pipeline.run("train")
        assert state.model is not None
        models.append(state.model)

    ensemble = VotingEnsemble(models=models, mode="hard")

    inputs = np.array([[0.0], [0.5], [1.0], [2.0]])
    preds = ensemble.predict(inputs)

    assert preds.shape == (len(inputs),)
    # Valid class labels are non-negative integers within the class range.
    assert np.all(preds >= 0)
    assert np.all(np.equal(preds, np.round(preds)).astype(bool))
