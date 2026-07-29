"""End-to-end integration test for the Phase 1 pipeline.

Tests the full pipeline lifecycle: data loading, model building,
training, evaluation, and export, using real Phase 1 components
with fake model/loss/optimizer.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from pipeline.config import Config
from pipeline.evaluation.metrics import Metrics, f1_score
from pipeline.hooks.progress import ProgressHook
from pipeline.pipeline import BasePipeline, PipelineState


def _make_tiny_csv(path: str) -> None:
    """Write a tiny CSV for E2E testing.

    Creates 8 rows with three feature columns and a binary label column.

    Args:
        path: Destination file path.
    """
    import pandas as pd

    df = pd.DataFrame({
        "f1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        "f2": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        "f3": [1, 0, 1, 0, 1, 0, 1, 0],
        "label": [0, 1, 0, 1, 0, 1, 0, 1],
    })
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


class TestPipelineE2E:
    """Full pipeline E2E with fake model/loss/optimizer."""

    def test_train_mode_completes(self) -> None:
        """Happy Path: run('train') completes all 6 stages.

        Verifies that:
        - history is populated with loss values per epoch
        - metrics are computed
        - predictions are populated
        - export file is written to disk
        """
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = f"{tmpdir}/train.csv"
            _make_tiny_csv(csv_path)

            config = Config(
                data_dir=tmpdir,
                output_dir=f"{tmpdir}/output",
                batch_size=2,
                num_epochs=2,
                train_ratio=0.75,
                seed=42,
            )

            from pipeline.evaluation.metrics import accuracy as acc_fn

            class E2EPipeline(BasePipeline):
                """Concrete pipeline for E2E testing with fake components."""

                def load_data(self, state: PipelineState) -> None:
                    """Stage 1: load CSV and split into train/val streams."""
                    from pipeline.data.csv_source import CsvDataSource
                    from pipeline.data.split import train_test_split

                    source = CsvDataSource(
                        csv_path, batch_size=config.batch_size, shuffle=False
                    )
                    train, val = train_test_split(
                        source, train_ratio=config.train_ratio
                    )
                    state.data_stream = train
                    state.val_data_stream = val

                def build_model(self, state: PipelineState) -> None:
                    """Stage 3: construct fake model, loss, optimizer, metrics."""
                    from pipeline.protocols import Loss

                    class _FakeModel:
                        """Minimal model returning random predictions."""

                        def forward(self, inputs: np.ndarray) -> np.ndarray:
                            return np.random.random(len(inputs))

                        def parameters(self) -> list:
                            return []

                        def train_mode(self) -> None:
                            pass

                        def eval_mode(self) -> None:
                            pass

                    class _FakeLossFn:
                        """MSE loss returning a Loss object."""

                        def forward(
                            self, p: np.ndarray, t: np.ndarray
                        ) -> Loss:
                            return Loss(
                                float(
                                    np.mean(
                                        (np.asarray(p) - np.asarray(t)) ** 2
                                    )
                                )
                            )

                        def __call__(
                            self, p: np.ndarray, t: np.ndarray
                        ) -> Loss:
                            return self.forward(p, t)

                    class _FakeOpt:
                        """No-op optimizer."""

                        def step(self) -> None:
                            pass

                        def zero_grad(self) -> None:
                            pass

                    state.model = _FakeModel()
                    state.loss_fn = _FakeLossFn()
                    state.optimizer = _FakeOpt()
                    state.metrics = Metrics(accuracy=acc_fn, f1=f1_score)

                def evaluate(self, state: PipelineState) -> None:
                    """Stage 5: evaluate on validation data."""
                    state.model.eval_mode()
                    all_preds, all_targets = [], []
                    for batch in state.val_data_stream:
                        preds = np.asarray(state.model.forward(batch.inputs))
                        all_preds.append(preds.reshape(-1))
                        all_targets.append(np.asarray(batch.targets))
                    y_pred = np.concatenate(all_preds)
                    y_true = np.concatenate(all_targets)
                    y_pred_binary = (y_pred > 0.5).astype(int)
                    state.metrics.compute(y_true, y_pred_binary)

                def export(self, state: PipelineState) -> None:
                    """Stage 6: write predictions to CSV."""
                    from pipeline.export.to_csv import to_csv

                    if state.val_data_stream is not None:
                        preds = []
                        for batch in state.val_data_stream:
                            p = np.asarray(
                                state.model.forward(batch.inputs)
                            )
                            preds.append(p.reshape(-1))
                        state.predictions = np.concatenate(preds)
                        # to_csv reshapes 1D arrays to (1, N);
                        # reshape to (N, 1) so columns match.
                        col_array = state.predictions.reshape(-1, 1)
                        to_csv(
                            col_array,
                            f"{config.output_dir}/predictions.csv",
                            columns=["prediction"],
                        )

            pipeline = E2EPipeline(config)
            pipeline.add_hook(ProgressHook())
            state = pipeline.run("train")

            # Verify state populated
            assert state.history is not None
            assert "loss" in state.history
            assert len(state.history["loss"]) == 2  # 2 epochs
            assert "accuracy" in state.metrics
            assert state.predictions is not None
            # Verify export file
            assert Path(f"{config.output_dir}/predictions.csv").exists()

    def test_infer_mode_skips_training(self) -> None:
        """Happy Path: run('infer') skips training stages.

        Verifies that in infer mode:
        - model is not built (None)
        - history is not populated (None)
        - current_epoch stays at 0
        - export still runs and predictions are available
        """
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = f"{tmpdir}/train.csv"
            _make_tiny_csv(csv_path)

            config = Config(
                data_dir=tmpdir,
                output_dir=f"{tmpdir}/output",
                batch_size=2,
            )

            class InferPipeline(BasePipeline):
                """Concrete pipeline for inference-only E2E test."""

                def load_data(self, state: PipelineState) -> None:
                    """Stage 1: load CSV and split."""
                    from pipeline.data.csv_source import CsvDataSource
                    from pipeline.data.split import train_test_split

                    source = CsvDataSource(
                        csv_path, batch_size=2, shuffle=False
                    )
                    train, val = train_test_split(source)
                    state.data_stream = train
                    state.val_data_stream = val

                def build_model(self, state: PipelineState) -> None:
                    """Stage 3: no-op in infer mode."""
                    pass

                def evaluate(self, state: PipelineState) -> None:
                    """Stage 5: no-op in infer mode."""
                    pass

                def export(self, state: PipelineState) -> None:
                    """Stage 6: write dummy predictions to CSV."""
                    from pipeline.export.to_csv import to_csv

                    state.predictions = np.array([0, 1])
                    to_csv(
                        state.predictions,
                        f"{config.output_dir}/predictions.csv",
                    )

            pipeline = InferPipeline(config)
            state = pipeline.run("infer")

            # In infer mode, training-related fields remain at defaults
            assert state.model is None
            assert state.history is None
            assert state.current_epoch == 0
            # But export still ran
            assert state.predictions is not None
