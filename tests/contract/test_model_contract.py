"""Contract tests: same tests run on SklearnModel and NumpyModel.

Verifies that every ModelProtocol implementation satisfies the
pipeline contract — forward shape, mode toggling, and end-to-end
on the Titanic dataset.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest

try:
    import sklearn  # noqa: F401

    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False

try:
    import torch  # noqa: F401

    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

if TYPE_CHECKING:
    from collections.abc import Callable


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_titanic_csv(path: str) -> str:
    """Write a small Titanic-style CSV for contract tests.

    Returns the file path.
    """
    import pandas as pd

    df = pd.DataFrame(
        {
            "feature_a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            "feature_b": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
            "feature_c": [22.0, 38.0, 25.0, 40.0, 30.0, 35.0, 28.0, 45.0],
            "survived": [0, 1, 0, 1, 0, 1, 0, 1],
        }
    )
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def _make_sklearn_pipeline(csv_path: str):
    """Build a pipeline with SklearnModel."""
    from sklearn.linear_model import LogisticRegression

    from pipeline.adapters.sklearn_adapter import SklearnModel, StubLoss, StubOptimizer
    from pipeline.config import Config
    from pipeline.data.csv_source import CsvDataSource
    from pipeline.data.split import train_test_split
    from pipeline.data.utils import collect_arrays
    from pipeline.evaluation.metrics import Metrics, accuracy
    from pipeline.pipeline import BasePipeline, PipelineState

    config = Config(
        batch_size=4,
        num_epochs=1,
        train_ratio=0.75,
        seed=42,
        output_dir=f"{Path(csv_path).parent}/output_sklearn",
    )

    class SklearnPipeline(BasePipeline):
        """Sklearn LogisticRegression on Titanic test subset."""

        def load_data(self, state: PipelineState) -> None:
            source = CsvDataSource(csv_path, batch_size=config.batch_size, shuffle=False)
            train, val = train_test_split(source, train_ratio=config.train_ratio)
            state.data_stream = train
            state.val_data_stream = val

        def build_model(self, state: PipelineState) -> None:
            state.model = SklearnModel(LogisticRegression(max_iter=1000))
            state.loss_fn = StubLoss()
            state.optimizer = StubOptimizer()
            state.metrics = Metrics(accuracy=accuracy)

        def train(self, state: PipelineState) -> None:
            X, y = collect_arrays(state.data_stream)
            state.model._estimator.fit(X, y)
            state.history = {"loss": [0.0]}

        def evaluate(self, state: PipelineState) -> None:
            state.model.eval_mode()
            all_preds, all_targets = [], []
            for batch in state.val_data_stream:
                probs = np.asarray(state.model.forward(batch.inputs))
                all_preds.append(np.argmax(probs, axis=1))
                all_targets.append(np.asarray(batch.targets))
            y_pred = np.concatenate(all_preds)
            y_true = np.concatenate(all_targets)
            state.metrics.compute(y_true, y_pred)

        def export(self, state: PipelineState) -> None:
            state.predictions = np.array([0])

    return SklearnPipeline(config)


def _make_numpy_pipeline(csv_path: str):
    """Build a pipeline with NumpyModel."""
    from pipeline.adapters.numpy_adapter import NumpyModel, NumpyOptimizer
    from pipeline.config import Config
    from pipeline.data.csv_source import CsvDataSource
    from pipeline.data.split import train_test_split
    from pipeline.evaluation.metrics import Metrics, accuracy
    from pipeline.pipeline import BasePipeline, PipelineState
    from pipeline.training.losses import CrossEntropyLoss
    from pipeline.training.optimizers import SGD

    config = Config(
        batch_size=4,
        num_epochs=5,
        train_ratio=0.75,
        learning_rate=0.5,
        seed=42,
        output_dir=f"{Path(csv_path).parent}/output_numpy",
    )

    class NumpyPipeline(BasePipeline):
        """Two-layer numpy MLP on Titanic test subset."""

        def load_data(self, state: PipelineState) -> None:
            source = CsvDataSource(csv_path, batch_size=config.batch_size, shuffle=False)
            train, val = train_test_split(source, train_ratio=config.train_ratio)
            state.data_stream = train
            state.val_data_stream = val

        def build_model(self, state: PipelineState) -> None:
            rng = np.random.default_rng(42)
            W1 = rng.standard_normal((8, 3)) * 0.1
            b1 = np.zeros(8)
            W2 = rng.standard_normal((2, 8)) * 0.1
            b2 = np.zeros(2)
            state.model = NumpyModel([(W1, b1), (W2, b2)], activation="relu")
            state.loss_fn = CrossEntropyLoss(model=state.model)
            state.optimizer = NumpyOptimizer(state.model.parameters(), SGD(lr=config.learning_rate))
            state.metrics = Metrics(accuracy=accuracy)

        def evaluate(self, state: PipelineState) -> None:
            state.model.eval_mode()
            all_preds, all_targets = [], []
            for batch in state.val_data_stream:
                logits = np.asarray(state.model.forward(batch.inputs))
                all_preds.append(np.argmax(logits, axis=1))
                all_targets.append(np.asarray(batch.targets))
            y_pred = np.concatenate(all_preds)
            y_true = np.concatenate(all_targets)
            state.metrics.compute(y_true, y_pred)

        def export(self, state: PipelineState) -> None:
            state.predictions = np.array([0])

    return NumpyPipeline(config)


def _make_torch_pipeline(csv_path: str):
    """Build a pipeline with TorchModel on tabular data."""
    import torch.nn as nn

    from pipeline.adapters.torch_adapter import TorchModel, TorchLoss, TorchOptimizer
    from pipeline.config import Config
    from pipeline.data.csv_source import CsvDataSource
    from pipeline.data.split import train_test_split
    from pipeline.evaluation.metrics import Metrics, accuracy
    from pipeline.pipeline import BasePipeline, PipelineState

    config = Config(
        batch_size=4,
        num_epochs=5,
        train_ratio=0.75,
        learning_rate=0.1,
        seed=42,
        output_dir=f"{Path(csv_path).parent}/output_torch",
    )

    class TorchPipeline(BasePipeline):
        """Torch Linear model on Titanic test subset."""

        def load_data(self, state: PipelineState) -> None:
            source = CsvDataSource(csv_path, batch_size=config.batch_size, shuffle=False)
            train, val = train_test_split(source, train_ratio=config.train_ratio)
            state.data_stream = train
            state.val_data_stream = val

        def build_model(self, state: PipelineState) -> None:
            module = nn.Linear(3, 2)
            state.model = TorchModel(module)
            state.loss_fn = TorchLoss("CrossEntropyLoss", model=state.model)
            state.optimizer = TorchOptimizer(
                state.model.parameters(), "SGD", lr=config.learning_rate
            )
            state.metrics = Metrics(accuracy=accuracy)

        def evaluate(self, state: PipelineState) -> None:
            state.model.eval_mode()
            all_preds, all_targets = [], []
            for batch in state.val_data_stream:
                logits = np.asarray(state.model.forward(batch.inputs))
                all_preds.append(np.argmax(logits, axis=1))
                all_targets.append(np.asarray(batch.targets))
            y_pred = np.concatenate(all_preds)
            y_true = np.concatenate(all_targets)
            state.metrics.compute(y_true, y_pred)

        def export(self, state: PipelineState) -> None:
            state.predictions = np.array([0])

    return TorchPipeline(config)


# ---------------------------------------------------------------------------
# Pipeline builders registry
# ---------------------------------------------------------------------------


def _get_pipeline_builders() -> dict[str, Callable]:
    """Return available pipeline builders, keyed by adapter name."""
    builders: dict[str, Callable] = {"numpy": _make_numpy_pipeline}
    if _HAS_SKLEARN:
        builders["sklearn"] = _make_sklearn_pipeline
    if _HAS_TORCH:
        builders["torch"] = _make_torch_pipeline
    return builders


# ---------------------------------------------------------------------------
# Contract tests
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestModelContract:
    """Every ModelProtocol adapter must pass these."""

    @pytest.mark.parametrize("adapter_name", list(_get_pipeline_builders().keys()))
    def test_forward_shape_matches_batch_size(self, adapter_name: str) -> None:
        """forward(inputs).shape[0] must equal the number of input samples."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = _make_titanic_csv(f"{tmpdir}/train.csv")
            builders = _get_pipeline_builders()
            pipeline = builders[adapter_name](csv_path)
            state = pipeline.run("train")
            model = state.model
            test_input = np.array([[1.0, 0.1, 22.0], [2.0, 0.2, 38.0]])
            output = model.forward(test_input)
            assert output.shape[0] == 2, (
                f"{adapter_name}: forward output batch size mismatch: "
                f"expected 2, got {output.shape[0]}"
            )

    @pytest.mark.parametrize("adapter_name", list(_get_pipeline_builders().keys()))
    def test_train_eval_mode_cycle_no_error(self, adapter_name: str) -> None:
        """train_mode() → eval_mode() → train_mode() must not raise."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = _make_titanic_csv(f"{tmpdir}/train.csv")
            builders = _get_pipeline_builders()
            pipeline = builders[adapter_name](csv_path)
            state = pipeline.run("train")
            model = state.model
            model.train_mode()
            model.eval_mode()
            model.train_mode()

    @pytest.mark.parametrize("adapter_name", list(_get_pipeline_builders().keys()))
    def test_full_pipeline_on_titanic(self, adapter_name: str) -> None:
        """Same Titanic task, both adapters complete all 6 stages."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = _make_titanic_csv(f"{tmpdir}/train.csv")
            builders = _get_pipeline_builders()
            pipeline = builders[adapter_name](csv_path)
            state = pipeline.run("train")
            # All stages completed
            assert state.history is not None, f"{adapter_name}: history not populated"
            assert "accuracy" in state.metrics, f"{adapter_name}: accuracy not computed"
            assert state.predictions is not None, f"{adapter_name}: predictions not set"
