"""Model checkpoint save/load — framework-specific serialization.

Provides :class:`TorchCheckpoint` for PyTorch model checkpoints.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pipeline.pipeline import PipelineState


class TorchCheckpoint:
    """Save and load PyTorch model checkpoints.

    Serializes model state_dict, optimizer state_dict, epoch, and training
    history using ``torch.save`` / ``torch.load``.

    Usage::

        TorchCheckpoint.save(state, "checkpoints/epoch_5.pt")
        TorchCheckpoint.load(state, "checkpoints/epoch_5.pt")
    """

    @staticmethod
    def save(state: PipelineState, path: str | Path) -> None:
        """Save model + optimizer + metadata to a checkpoint file.

        Args:
            state: PipelineState with model, optimizer, history, current_epoch.
            path: Output file path. Parent dirs created if needed.

        Raises:
            TypeError: If ``state.model`` is not a TorchModel.
        """
        from pipeline.adapters.torch_adapter import TorchModel

        if state.model is None or not isinstance(state.model, TorchModel):
            raise TypeError(
                "TorchCheckpoint.save expects state.model to be a TorchModel, "
                f"got {type(state.model).__name__ if state.model else 'None'}"
            )

        import torch  # type: ignore[import-not-found]

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint: dict[str, Any] = {
            "model_state_dict": state.model._module.state_dict(),
            "epoch": state.current_epoch,
        }
        if state.optimizer is not None:
            from pipeline.adapters.torch_adapter import TorchOptimizer

            if isinstance(state.optimizer, TorchOptimizer):
                checkpoint["optimizer_state_dict"] = state.optimizer._opt.state_dict()
        if state.history is not None:
            checkpoint["history"] = state.history

        torch.save(checkpoint, path)

    @staticmethod
    def load(state: PipelineState, path: str | Path) -> None:
        """Load model weights + optimizer state from a checkpoint file.

        Args:
            state: PipelineState with model (and optionally optimizer). The
                model must already be constructed (weights are loaded into it).
            path: Checkpoint file path.

        Raises:
            FileNotFoundError: If the checkpoint file doesn't exist.
            TypeError: If ``state.model`` is not a TorchModel.
        """
        import torch  # type: ignore[import-not-found]

        from pipeline.adapters.torch_adapter import TorchModel, TorchOptimizer

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        if state.model is None or not isinstance(state.model, TorchModel):
            raise TypeError(
                "TorchCheckpoint.load expects state.model to be a TorchModel, "
                f"got {type(state.model).__name__ if state.model else 'None'}"
            )

        checkpoint = torch.load(path, map_location="cpu", weights_only=True)

        state.model._module.load_state_dict(checkpoint["model_state_dict"])

        if (
            "optimizer_state_dict" in checkpoint
            and state.optimizer is not None
            and isinstance(state.optimizer, TorchOptimizer)
        ):
            state.optimizer._opt.load_state_dict(checkpoint["optimizer_state_dict"])

        state.current_epoch = checkpoint.get("epoch", state.current_epoch)

        if "history" in checkpoint and state.history is None:
            state.history = checkpoint["history"]
