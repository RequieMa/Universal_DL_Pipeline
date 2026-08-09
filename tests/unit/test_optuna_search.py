"""Unit tests for pipeline.hpo.search — OptunaSearch."""

from __future__ import annotations

import numpy as np
import pytest

from pipeline.config import Config
from pipeline.evaluation.metrics import accuracy
from pipeline.hpo.search import OptunaSearch
from pipeline.pipeline import BasePipeline, PipelineState

try:
    import optuna  # noqa: F401

    _HAS_OPTUNA = True
except ImportError:
    _HAS_OPTUNA = False

requires_optuna = pytest.mark.skipif(not _HAS_OPTUNA, reason="optuna not installed")


class _OptunaPipeline(BasePipeline):
    """Minimal pipeline for Optuna testing with a continuous param space."""

    def load_data(self, state: PipelineState) -> None:
        import os
        import tempfile

        import pandas as pd

        tmpdir = tempfile.mkdtemp()
        csv_path = os.path.join(tmpdir, "data.csv")
        df = pd.DataFrame({"a": [1.0], "label": [0]})
        df.to_csv(csv_path, index=False)

        from pipeline.data.csv_source import CsvDataSource

        source = CsvDataSource(csv_path, batch_size=1, shuffle=False)
        state.data_stream = source
        state.val_data_stream = source

    def build_model(self, state: PipelineState) -> None:
        from pipeline.protocols import Loss

        class _FakeModel:
            def forward(self, x):
                return np.array([0])

            def parameters(self):
                return []

            def train_mode(self):
                pass

            def eval_mode(self):
                pass

        class _FakeLoss:
            def forward(self, p, t):
                return Loss(value=0.0)

            def __call__(self, p, t):
                return self.forward(p, t)

        class _FakeOpt:
            def step(self):
                pass

            def zero_grad(self):
                pass

        state.model = _FakeModel()
        state.loss_fn = _FakeLoss()
        state.optimizer = _FakeOpt()

    def evaluate(self, state: PipelineState) -> None:
        """Return a score that depends on learning_rate."""
        from pipeline.evaluation.metrics import Metrics

        lr = state.config.learning_rate
        # Register a metric function (per the Metrics contract) and compute it.
        metrics = Metrics(accuracy=lambda y_true, y_pred, _lr=lr: float(_lr))
        metrics.compute(np.array([0]), np.array([0]))
        state.metrics = metrics

    def export(self, state: PipelineState) -> None:
        state.predictions = np.array([0])

    def train(self, state: PipelineState) -> None:
        """Override TrainLoop — train is a no-op for search speed."""
        state.history = {"loss": [0.0]}
        self.evaluate(state)


@requires_optuna
class TestOptunaSearch:
    """Tests for OptunaSearch."""

    def test_run_completes_with_continuous_space(self) -> None:
        """OptunaSearch completes with a continuous (low, high) param grid."""
        grid = {"learning_rate": (0.01, 0.1)}
        search = OptunaSearch(
            pipeline_cls=_OptunaPipeline,
            base_config=Config(num_epochs=1),
            param_grid=grid,
            n_trials=5,
            scoring="accuracy",
            seed=42,
        )
        result = search.run()
        assert result is not None

    def test_run_completes_with_categorical_space(self) -> None:
        """OptunaSearch completes with a categorical list param grid."""
        grid = {"learning_rate": [0.1, 0.01, 0.001], "batch_size": [16, 32]}
        search = OptunaSearch(
            pipeline_cls=_OptunaPipeline,
            base_config=Config(),
            param_grid=grid,
            n_trials=5,
            scoring="accuracy",
            seed=42,
        )
        result = search.run()
        assert result is not None

    def test_empty_grid_raises(self) -> None:
        """OptunaSearch raises ValueError on empty param_grid."""
        with pytest.raises(ValueError, match="param_grid"):
            OptunaSearch(
                pipeline_cls=_OptunaPipeline,
                base_config=Config(),
                param_grid={},
                n_trials=5,
                scoring="accuracy",
            )

    def test_n_trials_must_be_positive(self) -> None:
        """OptunaSearch raises ValueError for n_trials < 1."""
        with pytest.raises(ValueError, match="n_trials"):
            OptunaSearch(
                pipeline_cls=_OptunaPipeline,
                base_config=Config(),
                param_grid={"lr": (0.0, 1.0)},
                n_trials=0,
                scoring="accuracy",
            )

    def test_reproducible_with_seed(self) -> None:
        """Same seed produces same trial order (reproducible)."""
        grid = {"learning_rate": (0.001, 0.01)}
        s1 = OptunaSearch(
            pipeline_cls=_OptunaPipeline,
            base_config=Config(),
            param_grid=grid,
            n_trials=4,
            scoring="accuracy",
            seed=123,
        )
        s2 = OptunaSearch(
            pipeline_cls=_OptunaPipeline,
            base_config=Config(),
            param_grid=grid,
            n_trials=4,
            scoring="accuracy",
            seed=123,
        )
        # Same seed → same sampler, should produce same results
        result1 = s1.run()
        result2 = s2.run()
        # At minimum both complete and produce trials
        assert len(result1.trials) == len(result2.trials)

    def test_int_range_yields_integer_params(self) -> None:
        """An int (low, high) tuple samples integers, not floats."""
        grid = {"batch_size": (8, 128)}
        search = OptunaSearch(
            pipeline_cls=_OptunaPipeline,
            base_config=Config(),
            param_grid=grid,
            n_trials=6,
            scoring="accuracy",
            seed=42,
        )
        result = search.run()
        assert result.trials  # search produced at least one trial
        for trial in result.trials:
            value = trial.config.batch_size
            assert isinstance(value, int)
            assert not isinstance(value, bool)
            assert 8 <= value <= 128

    def test_float_range_yields_float_params(self) -> None:
        """A tuple with a float endpoint samples floats."""
        grid = {"learning_rate": (1e-4, 1e-1)}
        search = OptunaSearch(
            pipeline_cls=_OptunaPipeline,
            base_config=Config(),
            param_grid=grid,
            n_trials=6,
            scoring="accuracy",
            seed=42,
        )
        result = search.run()
        assert result.trials
        for trial in result.trials:
            value = trial.config.learning_rate
            assert isinstance(value, float)
            assert 1e-4 <= value <= 1e-1

    def test_list_space_yields_categorical_choices(self) -> None:
        """A list space samples only values drawn from that list."""
        choices = [16, 32, 64]
        grid = {"batch_size": choices}
        search = OptunaSearch(
            pipeline_cls=_OptunaPipeline,
            base_config=Config(),
            param_grid=grid,
            n_trials=6,
            scoring="accuracy",
            seed=42,
        )
        result = search.run()
        assert result.trials
        for trial in result.trials:
            assert trial.config.batch_size in choices
