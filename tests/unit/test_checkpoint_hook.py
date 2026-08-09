"""Unit tests for pipeline.hooks.checkpoint — CheckpointHook."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

try:
    import torch  # noqa: F401  # availability probe; only torch.nn is imported below

    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

requires_torch = pytest.mark.skipif(not _HAS_TORCH, reason="torch not installed")


@requires_torch
class TestCheckpointHook:
    """Tests for CheckpointHook."""

    def test_saves_on_epoch_end(self, tmp_path: Path) -> None:
        """CheckpointHook saves a checkpoint at on_epoch_end."""
        import torch.nn as nn

        from pipeline.adapters.torch_adapter import TorchModel, TorchOptimizer
        from pipeline.config import Config
        from pipeline.hooks.checkpoint import CheckpointHook
        from pipeline.pipeline import PipelineState

        module = nn.Linear(2, 1)
        state = PipelineState(
            config=Config(output_dir=str(tmp_path), checkpoint_dir=str(tmp_path / "ckpt"))
        )
        state.model = TorchModel(module)
        state.optimizer = TorchOptimizer(state.model.parameters(), "SGD", lr=0.01)
        state.history = {"loss": [0.5, 0.3]}

        hook = CheckpointHook()
        hook.on_epoch_end(0, state)

        assert (tmp_path / "ckpt" / "latest.pt").exists()
        assert (tmp_path / "ckpt" / "best.pt").exists()  # first epoch always saves best

    def test_best_only_mode(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """save_best_only=True writes best.pt only when the epoch metric improves.

        Asserts on the save *decision* (via a spy that counts writes) rather
        than file mtimes, which can collide within a single clock tick and
        make the test flaky.
        """
        import torch.nn as nn

        from pipeline.adapters.torch_adapter import TorchModel, TorchOptimizer
        from pipeline.config import Config
        from pipeline.export import checkpoint as checkpoint_mod
        from pipeline.hooks.checkpoint import CheckpointHook
        from pipeline.pipeline import PipelineState

        # Spy: record every path passed to TorchCheckpoint.save without
        # touching the disk, so we can assert on writes deterministically.
        saved_paths: list[Path] = []

        def fake_save(state: PipelineState, path: Path) -> None:
            saved_paths.append(Path(path))

        monkeypatch.setattr(checkpoint_mod.TorchCheckpoint, "save", staticmethod(fake_save))

        module = nn.Linear(2, 1)
        state = PipelineState(
            config=Config(output_dir=str(tmp_path), checkpoint_dir=str(tmp_path / "ckpt"))
        )
        state.model = TorchModel(module)
        state.optimizer = TorchOptimizer(state.model.parameters(), "SGD", lr=0.01)
        state.history = {"loss": [0.3]}

        hook = CheckpointHook(save_best_only=True)
        best_path = tmp_path / "ckpt" / "best.pt"

        # Epoch 0: mean 0.3 -> first epoch always saves best.
        hook.on_epoch_end(0, state)
        assert saved_paths == [best_path]

        # Epoch 1: mean 0.5 -> worse, must NOT write again.
        state.history["loss"].append(0.5)
        hook.on_epoch_end(1, state)
        assert saved_paths == [best_path]  # no second write on non-improving epoch

        # Epoch 2: mean 0.2 -> better, SHOULD write again.
        state.history["loss"].append(0.2)
        hook.on_epoch_end(2, state)
        assert saved_paths == [best_path, best_path]  # improving epoch rewrites best

    def test_skips_when_model_is_none(self, tmp_path: Path) -> None:
        """on_epoch_end is no-op when state.model is None."""
        from pipeline.config import Config
        from pipeline.hooks.checkpoint import CheckpointHook
        from pipeline.pipeline import PipelineState

        state = PipelineState(config=Config(checkpoint_dir=str(tmp_path)))
        state.model = None

        hook = CheckpointHook()
        hook.on_epoch_end(0, state)  # should not raise

    def test_skips_when_history_is_none(self, tmp_path: Path) -> None:
        """on_epoch_end is no-op when state.history is None."""
        import torch.nn as nn

        from pipeline.adapters.torch_adapter import TorchModel, TorchOptimizer
        from pipeline.config import Config
        from pipeline.hooks.checkpoint import CheckpointHook
        from pipeline.pipeline import PipelineState

        module = nn.Linear(2, 1)
        state = PipelineState(config=Config(output_dir=str(tmp_path)))
        state.model = TorchModel(module)
        state.optimizer = TorchOptimizer(state.model.parameters(), "SGD", lr=0.01)
        state.history = None

        hook = CheckpointHook()
        hook.on_epoch_end(0, state)  # should not raise


class TestCheckpointHookLogic:
    """Torch-free tests for CheckpointHook's decision logic.

    These spy on ``TorchCheckpoint.save`` so no torch import is triggered:
    the hook only imports torch lazily inside ``save``. A plain sentinel is
    used for ``state.model`` (just needs to be non-None).
    """

    def _spy_state(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, history: dict
    ) -> tuple:
        from pipeline.config import Config
        from pipeline.export import checkpoint as checkpoint_mod
        from pipeline.pipeline import PipelineState

        saved_paths: list[Path] = []

        def fake_save(state: PipelineState, path: Path) -> None:
            saved_paths.append(Path(path))

        monkeypatch.setattr(checkpoint_mod.TorchCheckpoint, "save", staticmethod(fake_save))

        state = PipelineState(
            config=Config(output_dir=str(tmp_path), checkpoint_dir=str(tmp_path / "ckpt"))
        )
        state.model = object()  # sentinel: only needs to be non-None
        state.history = history
        return state, saved_paths

    def test_best_tracks_epoch_mean_not_last_batch(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """best.pt tracks the epoch MEAN, not the last batch.

        Flat per-batch history:
          - epoch 0 batches [0.0, 1.0] -> mean 0.5 (first epoch: saves best)
          - epoch 1 batches [1.0, 0.0] -> mean 0.5, last 0.0
        Last-batch logic (0.0 < 0.5) would rewrite best; mean logic
        (0.5 not < 0.5) must NOT.
        """
        state, saved_paths = self._spy_state(tmp_path, monkeypatch, {"loss": []})
        from pipeline.hooks.checkpoint import CheckpointHook

        hook = CheckpointHook(save_best_only=True)
        best_path = tmp_path / "ckpt" / "best.pt"

        state.history["loss"].extend([0.0, 1.0])  # mean 0.5
        hook.on_epoch_end(0, state)
        assert saved_paths == [best_path]

        state.history["loss"].extend([1.0, 0.0])  # mean 0.5, last 0.0
        hook.on_epoch_end(1, state)
        assert saved_paths == [best_path]  # mean did not improve -> no rewrite

    def test_missing_key_warns_once_and_no_save(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog
    ) -> None:
        """A missing monitored key logs one warning and never writes best.pt."""
        state, saved_paths = self._spy_state(tmp_path, monkeypatch, {"loss": [0.5]})
        from pipeline.hooks.checkpoint import CheckpointHook

        hook = CheckpointHook(save_best_only=True, monitor="val_loss")

        with caplog.at_level(logging.WARNING, logger="pipeline.hooks.checkpoint"):
            hook.on_epoch_end(0, state)
            state.history["loss"].append(0.4)
            hook.on_epoch_end(1, state)

        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1
        assert "val_loss" in warnings[0].getMessage()
        assert saved_paths == []  # missing metric never counts as improvement
