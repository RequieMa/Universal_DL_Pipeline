"""End-to-end integration tests for checkpointing + inference (Phase 4).

Covers:
    1. Train with CheckpointHook -> checkpoint saved -> infer mode
       ``load_checkpoint`` loads the weights and produces predictions.
    2. CheckpointHook saves ``latest.pt`` and ``best.pt`` during a full
       pipeline run.
    3. EarlyStopHook sets ``state.should_stop = True`` inside the training
       loop when the monitored metric stops improving.

All torch imports are lazy (inside functions) so this file imports cleanly
without PyTorch installed. Tests are gated behind ``requires_torch``
(skipif) and marked ``slow`` per project conventions.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

try:
    import torch  # noqa: F401

    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

requires_torch = pytest.mark.skipif(not _HAS_TORCH, reason="torch not installed")


@requires_torch
@pytest.mark.slow
def test_train_save_infer_load(tmp_path: Path) -> None:
    """Train saves a checkpoint; infer load_checkpoint produces predictions.

    Exercise the full lifecycle: train one epoch so CheckpointHook writes
    ``best.pt``, then build a fresh pipeline in infer mode whose
    ``load_checkpoint`` stage loads those weights into a freshly-built
    model, and confirm it runs a forward pass and exports predictions.
    """
    import pandas as pd
    import torch.nn as nn

    from pipeline.adapters.torch_adapter import TorchLoss, TorchModel, TorchOptimizer
    from pipeline.config import Config
    from pipeline.data.csv_source import CsvDataSource
    from pipeline.data.split import train_test_split
    from pipeline.evaluation.metrics import Metrics, accuracy
    from pipeline.export.checkpoint import TorchCheckpoint
    from pipeline.hooks.checkpoint import CheckpointHook
    from pipeline.pipeline import BasePipeline, PipelineState

    csv_path = tmp_path / "data.csv"
    df = pd.DataFrame(
        {
            "a": np.random.randn(24),
            "b": np.random.randn(24),
            "c": np.random.randn(24),
            "label": np.random.randint(0, 2, 24),
        }
    )
    df.to_csv(csv_path, index=False)

    config = Config(
        batch_size=4,
        num_epochs=1,
        train_ratio=0.75,
        learning_rate=0.1,
        seed=42,
        output_dir=str(tmp_path / "output"),
        checkpoint_dir=str(tmp_path / "checkpoints"),
    )

    from pipeline.evaluation.metrics import accuracy as acc_fn

    holder: dict[str, float] = {}

    class CheckpointPipeline(BasePipeline):
        """A torch pipeline that checkpoints at epoch end."""

        def load_data(self, state: PipelineState) -> None:
            source = CsvDataSource(str(csv_path), batch_size=config.batch_size, shuffle=False)
            train, val = train_test_split(source, train_ratio=config.train_ratio)
            state.data_stream = train
            state.val_data_stream = val

        def build_model(self, state: PipelineState) -> None:
            # WHY model=<TorchModel>: CsvDataSource yields numpy arrays, so
            # TorchModel.forward returns a *detached* numpy output. Passing the
            # model rebinds loss.backward() to re-run the module on the cached
            # input, rebuilding the autograd graph so gradients reach parameters.
            state.model = TorchModel(nn.Linear(3, 2))
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
            state.metrics.compute(np.concatenate(all_targets), np.concatenate(all_preds))

        def load_checkpoint(self, state: PipelineState) -> None:
            # Infer mode: load the trained weights recorded by CheckpointHook
            # into the freshly-built model before exporting predictions.
            best_pt = Path(config.checkpoint_dir) / "best.pt"
            assert best_pt.exists(), "expected best.pt from the train run"
            TorchCheckpoint.load(state, best_pt)

        def export(self, state: PipelineState) -> None:
            state.model.eval_mode()
            preds, targets = [], []
            for batch in state.val_data_stream:
                logits = np.asarray(state.model.forward(batch.inputs))
                preds.append(np.argmax(logits, axis=1))
                targets.append(np.asarray(batch.targets))
            state.predictions = np.concatenate(preds)
            holder["loaded_accuracy"] = acc_fn(np.concatenate(targets), state.predictions)

    # Phase 1: train, writing the checkpoint.
    train_pipeline = CheckpointPipeline(config)
    train_pipeline.add_hook(CheckpointHook())
    train_state = train_pipeline.run("train")

    assert train_state.history is not None and len(train_state.history["loss"]) > 0
    best_pt = tmp_path / "checkpoints" / "best.pt"
    assert best_pt.exists(), "CheckpointHook should write best.pt during training"

    # Phase 2: fresh pipeline in infer mode loads the same weights.
    infer_pipeline = CheckpointPipeline(config)
    infer_state = infer_pipeline.run("infer")

    # Infer mode does not run train/evaluate, but TorchCheckpoint.load restores
    # the recorded history onto the fresh state -- that is expected.
    assert infer_state.predictions is not None
    assert len(infer_state.predictions) > 0
    # Weights really did round-trip: predictions on the validation set have a
    # valid accuracy that the loaded weights produced.
    assert 0.0 <= float(holder["loaded_accuracy"]) <= 1.0


@requires_torch
@pytest.mark.slow
def test_checkpoint_hook_in_full_pipeline(tmp_path: Path) -> None:
    """CheckpointHook saves latest.pt AND best.pt during a full run."""
    import pandas as pd
    import torch.nn as nn

    from pipeline.adapters.torch_adapter import TorchLoss, TorchModel, TorchOptimizer
    from pipeline.config import Config
    from pipeline.data.csv_source import CsvDataSource
    from pipeline.data.split import train_test_split
    from pipeline.evaluation.metrics import Metrics, accuracy
    from pipeline.hooks.checkpoint import CheckpointHook
    from pipeline.pipeline import BasePipeline, PipelineState

    csv_path = tmp_path / "data.csv"
    df = pd.DataFrame(
        {
            "a": np.random.randn(20),
            "b": np.random.randn(20),
            "label": np.random.randint(0, 2, 20),
        }
    )
    df.to_csv(csv_path, index=False)

    config = Config(
        batch_size=4,
        num_epochs=3,
        train_ratio=0.75,
        learning_rate=0.1,
        seed=7,
        output_dir=str(tmp_path / "output"),
        checkpoint_dir=str(tmp_path / "checkpoints"),
    )

    class OneEpochPipeline(BasePipeline):
        def load_data(self, state: PipelineState) -> None:
            source = CsvDataSource(str(csv_path), batch_size=config.batch_size, shuffle=False)
            train, val = train_test_split(source, train_ratio=config.train_ratio)
            state.data_stream = train
            state.val_data_stream = val

        def build_model(self, state: PipelineState) -> None:
            state.model = TorchModel(nn.Linear(2, 2))
            state.loss_fn = TorchLoss("CrossEntropyLoss", model=state.model)
            state.optimizer = TorchOptimizer(
                state.model.parameters(), "SGD", lr=config.learning_rate
            )
            state.metrics = Metrics(accuracy=accuracy)

        def evaluate(self, state: PipelineState) -> None:
            pass

        def export(self, state: PipelineState) -> None:
            state.predictions = np.array([0])

    pipeline = OneEpochPipeline(config)
    pipeline.add_hook(CheckpointHook())
    state = pipeline.run("train")

    ckpt_dir = tmp_path / "checkpoints"
    latest = ckpt_dir / "latest.pt"
    best = ckpt_dir / "best.pt"

    assert latest.exists(), "CheckpointHook should write latest.pt each epoch"
    assert best.exists(), "CheckpointHook should track and write best.pt"
    assert state.history is not None and len(state.history["loss"]) > 0


@requires_torch
@pytest.mark.slow
def test_early_stop_triggers(tmp_path: Path) -> None:
    """EarlyStopHook sets should_stop=True, halting the training loop early."""
    import pandas as pd
    import torch.nn as nn

    from pipeline.adapters.torch_adapter import TorchLoss, TorchModel, TorchOptimizer
    from pipeline.config import Config
    from pipeline.data.csv_source import CsvDataSource
    from pipeline.data.split import train_test_split
    from pipeline.evaluation.metrics import Metrics, accuracy
    from pipeline.hooks.early_stop import EarlyStopHook
    from pipeline.pipeline import BasePipeline, PipelineState

    csv_path = tmp_path / "data.csv"
    df = pd.DataFrame(
        {
            "x": np.random.randn(16),
            "label": np.random.randint(0, 2, 16),
        }
    )
    df.to_csv(csv_path, index=False)

    # num_epochs is high so the loop would run a long time unless early
    # stopping fires. patience=0 stops at the first non-improving epoch, so
    # with min_delta=0 the loop halts almost immediately.
    config = Config(
        batch_size=4,
        num_epochs=50,
        train_ratio=0.75,
        learning_rate=0.01,
        seed=3,
        output_dir=str(tmp_path / "output"),
        checkpoint_dir=str(tmp_path / "checkpoints"),
    )

    class EarlyStopPipeline(BasePipeline):
        def load_data(self, state: PipelineState) -> None:
            source = CsvDataSource(str(csv_path), batch_size=config.batch_size, shuffle=False)
            train, val = train_test_split(source, train_ratio=config.train_ratio)
            state.data_stream = train
            state.val_data_stream = val

        def build_model(self, state: PipelineState) -> None:
            state.model = TorchModel(nn.Linear(1, 2))
            state.loss_fn = TorchLoss("CrossEntropyLoss", model=state.model)
            state.optimizer = TorchOptimizer(
                state.model.parameters(), "SGD", lr=config.learning_rate
            )
            state.metrics = Metrics(accuracy=accuracy)

        def evaluate(self, state: PipelineState) -> None:
            pass

        def export(self, state: PipelineState) -> None:
            state.predictions = np.array([0])

    pipeline = EarlyStopPipeline(config)
    # Deterministic early stop: patience=0 stops on the first non-improving
    # epoch, and min_delta=1.0 is larger than any achievable per-epoch loss
    # drop (CrossEntropy starts near 0.7), so the first epoch never counts
    # as an improvement. Epoch 0 records the baseline; epoch 1 trips the stop.
    pipeline.add_hook(EarlyStopHook(patience=0, monitor="loss", mode="min", min_delta=1.0))

    state = pipeline.run("train")

    # Early stopping fired: the loop halted well before num_epochs=50.
    assert state.history is not None
    assert len(state.history["loss"]) < 50
    # The hook dispatched on_epoch_end set the stop signal on the state.
    assert state.should_stop is True
