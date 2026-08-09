"""Unit tests for the hyperparameter search framework (pipeline.hpo).

Tests cover :class:`TrialResult`, :class:`SearchResult`, :class:`GridSearch`,
and :class:`RandomSearch` using a minimal fake pipeline that never touches a
real framework or dataset.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from pipeline.config import Config
from pipeline.evaluation import Metrics
from pipeline.hpo import (
    GridSearch,
    RandomSearch,
    SearchResult,
    TrialResult,
)
from pipeline.pipeline import BasePipeline
from pipeline.protocols import Loss


class _FakePipeline(BasePipeline):
    """Minimal pipeline whose score is a function of ``learning_rate``.

    ``evaluate`` writes ``accuracy = 0.5 + learning_rate * 5`` into
    ``state.metrics`` so tests can assert the search picked the config with
    the largest learning rate.
    """

    def load_data(self, state) -> None:
        """No-op data loader — the pipeline runs with no real data."""
        return None

    def train(self, state) -> None:
        """No-op training — scoring is deterministic and framework-free."""
        return None

    def build_model(self, state) -> None:
        """Construct trivial model, loss, and optimizer stubs."""

        class _FakeModel:
            def forward(self, x):
                return np.array([0])

            def parameters(self):
                return []

            def train_mode(self) -> None:
                return None

            def eval_mode(self) -> None:
                return None

        class _FakeLoss:
            def forward(self, p, t):
                return Loss(value=0.0)

            def __call__(self, p, t):
                return self.forward(p, t)

        class _FakeOpt:
            def step(self) -> None:
                return None

            def zero_grad(self) -> None:
                return None

        state.model = _FakeModel()
        state.loss_fn = _FakeLoss()
        state.optimizer = _FakeOpt()

    def evaluate(self, state) -> None:
        """Score the config: accuracy grows with learning_rate."""
        state.metrics = Metrics(accuracy=lambda y, p: float(np.mean(y == p)))
        state.metrics.compute(np.array([0]), np.array([0]))
        state.metrics._values["accuracy"] = 0.5 + state.config.learning_rate * 5

    def export(self, state) -> None:
        """Write a trivial prediction payload."""
        state.predictions = np.array([0])


def _param_grid() -> dict[str, list]:
    """A two-axis grid: 2 learning rates x 2 batch sizes = 4 combinations."""
    return {"learning_rate": [0.001, 0.01], "batch_size": [16, 32]}


# ── TrialResult ─────────────────────────────────────────────────────────


class TestTrialResult:
    """Construction and attribute access of a single trial."""

    def test_constructs_with_defaults(self) -> None:
        """Defaults produce an empty metrics dict and a None score."""
        config = Config()
        trial = TrialResult(trial_id=0, config=config)

        assert trial.trial_id == 0
        assert trial.config is config
        assert trial.metrics == {}
        assert trial.score is None

    def test_constructs_with_values(self) -> None:
        """Explicit metrics and score are stored."""
        config = Config()
        trial = TrialResult(
            trial_id=2,
            config=config,
            metrics={"accuracy": 0.9},
            score=0.9,
        )

        assert trial.trial_id == 2
        assert trial.metrics == {"accuracy": 0.9}
        assert trial.score == 0.9


# ── SearchResult ────────────────────────────────────────────────────────


class TestSearchResult:
    """Ranking and aggregation across trials."""

    def _trials(self) -> list[TrialResult]:
        """Three trials with scores 0.3, 0.9, 0.6 (best is trial 1)."""
        objs = [SimpleNamespace(mark=n) for n in (1, 2, 3)]
        return [
            TrialResult(trial_id=0, config=objs[0], metrics={"accuracy": 0.3}, score=0.3),
            TrialResult(trial_id=1, config=objs[1], metrics={"accuracy": 0.9}, score=0.9),
            TrialResult(trial_id=2, config=objs[2], metrics={"accuracy": 0.6}, score=0.6),
        ]

    def test_best_trial_is_highest_score(self) -> None:
        """best_trial returns the trial with the greatest score."""
        result = SearchResult(trials=self._trials(), scoring="accuracy")

        assert result.best_trial.trial_id == 1

    def test_best_config_and_score(self) -> None:
        """best_config and best_score mirror the best trial."""
        result = SearchResult(trials=self._trials(), scoring="accuracy")

        assert result.best_config.mark == 2
        assert result.best_score == 0.9

    def test_no_successful_trials(self) -> None:
        """None of the best_* properties raise when every trial failed."""
        trials = [
            TrialResult(trial_id=0, config=Config(), score=None),
            TrialResult(trial_id=1, config=Config(), score=None),
        ]
        result = SearchResult(trials=trials, scoring="accuracy")

        assert result.best_trial is None
        assert result.best_config is None
        assert result.best_score is None

    def test_none_scores_ignored_when_ranking(self) -> None:
        """Failed trials (None score) never become the best trial."""
        trials = [
            TrialResult(trial_id=0, config=Config(), score=None),
            TrialResult(trial_id=1, config=Config(), score=0.4),
            TrialResult(trial_id=2, config=Config(), score=None),
        ]
        result = SearchResult(trials=trials, scoring="accuracy")

        assert result.best_trial.trial_id == 1

    def test_to_dataframe_columns_and_rows(self) -> None:
        """to_dataframe exposes one row per trial plus config columns."""
        trials = [
            TrialResult(
                trial_id=0,
                config=Config(learning_rate=0.1, batch_size=16),
                metrics={"accuracy": 0.9},
                score=0.9,
            ),
            TrialResult(
                trial_id=1,
                config=Config(learning_rate=0.2, batch_size=32),
                metrics={"accuracy": 0.5},
                score=0.5,
            ),
        ]
        result = SearchResult(trials=trials, scoring="accuracy")

        df = result.to_dataframe()

        assert {"trial_id", "score", "learning_rate", "batch_size"} <= set(df.columns)
        assert len(df) == 2
        assert df.loc[0, "trial_id"] == 0


# ── GridSearch ──────────────────────────────────────────────────────────


class TestGridSearch:
    """Exhaustive Cartesian-product search."""

    def test_cartesian_product_count_is_four(self) -> None:
        """A 2x2 grid yields exactly 4 configurations."""
        search = GridSearch(
            param_grid=_param_grid(),
            pipeline_cls=_FakePipeline,
            base_config=Config(),
            scoring="accuracy",
        )

        configs = list(search._generate_configs())

        assert len(configs) == 4

    def test_cartesian_product_covers_all_combinations(self) -> None:
        """Every (lr, batch) pair from the grid is generated exactly once."""
        search = GridSearch(
            param_grid=_param_grid(),
            pipeline_cls=_FakePipeline,
            base_config=Config(),
            scoring="accuracy",
        )

        pairs = {(c.learning_rate, c.batch_size) for c in search._generate_configs()}

        assert pairs == {(0.001, 16), (0.001, 32), (0.01, 16), (0.01, 32)}

    def test_generated_configs_are_independent_copies(self) -> None:
        """Mutating one generated config must not affect the base config."""
        search = GridSearch(
            param_grid=_param_grid(),
            pipeline_cls=_FakePipeline,
            base_config=Config(learning_rate=0.001),
            scoring="accuracy",
        )

        configs = list(search._generate_configs())
        configs[0].learning_rate = 999.0

        assert all(c.learning_rate != 999.0 for c in configs[1:])
        assert search.base_config.learning_rate == 0.001

    def test_run_completes_n_trials(self) -> None:
        """run() evaluates every grid cell and ranks the results."""
        search = GridSearch(
            param_grid=_param_grid(),
            pipeline_cls=_FakePipeline,
            base_config=Config(),
            scoring="accuracy",
        )

        result = search.run()

        assert len(result.trials) == 4
        # accuracy = 0.5 + lr*5 → highest lr (0.01) wins
        assert result.best_config.learning_rate == 0.01
        assert result.best_score == pytest.approx(0.5 + 0.01 * 5)

    def test_empty_grid_raises(self) -> None:
        """An empty param_grid raises ValueError at construction."""
        with pytest.raises(ValueError):
            GridSearch(
                param_grid={},
                pipeline_cls=_FakePipeline,
                base_config=Config(),
                scoring="accuracy",
            )

    def test_failed_trial_is_skipped_and_logged(self) -> None:
        """A pipeline that raises is caught; the trial keeps a None score."""

        class _BoomPipeline(_FakePipeline):
            def evaluate(self, state) -> None:
                raise RuntimeError("boom")

        search = GridSearch(
            param_grid=_param_grid(),
            pipeline_cls=_BoomPipeline,
            base_config=Config(),
            scoring="accuracy",
        )

        result = search.run()

        assert len(result.trials) == 4
        assert all(t.score is None for t in result.trials)
        assert result.best_trial is None


# ── RandomSearch ────────────────────────────────────────────────────────


class TestRandomSearch:
    """Random sampling without replacement."""

    def test_n_trials_sampled(self) -> None:
        """With a 2x2 grid and n_trials=4, all 4 combinations are produced."""
        search = RandomSearch(
            param_grid=_param_grid(),
            n_trials=4,
            seed=42,
            pipeline_cls=_FakePipeline,
            base_config=Config(),
            scoring="accuracy",
        )

        configs = list(search._generate_configs())

        assert len(configs) == 4

    def test_same_seed_is_reproducible(self) -> None:
        """Two searches with the same seed yield the same configurations."""
        def configs_for(seed: int) -> list[float]:
            search = RandomSearch(
                param_grid=_param_grid(),
                n_trials=3,
                seed=seed,
                pipeline_cls=_FakePipeline,
                base_config=Config(),
                scoring="accuracy",
            )
            return [c.learning_rate for c in search._generate_configs()]

        assert configs_for(7) == configs_for(7)
        assert configs_for(7) != configs_for(8)

    def test_without_replacement(self) -> None:
        """Sampling with n_trials == total yields every distinct combination."""
        search = RandomSearch(
            param_grid=_param_grid(),
            n_trials=4,
            seed=42,
            pipeline_cls=_FakePipeline,
            base_config=Config(),
            scoring="accuracy",
        )

        pairs = {(c.learning_rate, c.batch_size) for c in search._generate_configs()}

        assert pairs == {(0.001, 16), (0.001, 32), (0.01, 16), (0.01, 32)}

    def test_clamp_n_trials(self) -> None:
        """n_trials > total combinations clamps down to the total."""
        search = RandomSearch(
            param_grid=_param_grid(),
            n_trials=100,
            seed=1,
            pipeline_cls=_FakePipeline,
            base_config=Config(),
            scoring="accuracy",
        )

        configs = list(search._generate_configs())

        assert len(configs) == 4

    def test_run_returns_valid_result(self) -> None:
        """run() produces a SearchResult with scores from the fake pipeline."""
        search = RandomSearch(
            param_grid={"learning_rate": [0.001, 0.01, 0.1]},
            n_trials=3,
            seed=0,
            pipeline_cls=_FakePipeline,
            base_config=Config(),
            scoring="accuracy",
        )

        result = search.run()

        assert len(result.trials) == 3
        assert all(t.score is not None for t in result.trials)
        # The fake pipeline always reaches best_score for lr = 0.1
        assert result.best_config.learning_rate == 0.1
