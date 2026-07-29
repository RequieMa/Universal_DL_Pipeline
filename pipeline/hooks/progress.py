"""Progress bar hook using tqdm.

Provides :class:`ProgressHook` — a concrete :class:`BaseHook`
that renders training progress with tqdm.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from pipeline.hooks.base import BaseHook

if TYPE_CHECKING:
    from pipeline.pipeline import PipelineState


class ProgressHook(BaseHook):
    """tqdm progress bar for training.

    Creates a progress bar on ``on_epoch_start``, updates it on
    ``on_batch_end`` with the current batch loss, and closes it
    on ``on_epoch_end``.

    Usage::

        pipeline.add_hook(ProgressHook())
    """

    def __init__(self) -> None:
        """Initialize the progress hook."""
        self._pbar = None
        self._losses: list[float] = []

    def on_epoch_start(
        self, epoch: int, state: PipelineState
    ) -> None:
        """Create a new tqdm progress bar for this epoch.

        Args:
            epoch: Zero-based epoch index.
            state: Current pipeline state.
        """
        from tqdm import tqdm

        total = len(state.data_stream) if state.data_stream is not None else 0
        self._pbar = tqdm(total=total, desc=f"Epoch {epoch + 1}")
        self._losses.clear()

    def on_batch_end(
        self, batch: int, loss: float, state: PipelineState
    ) -> None:
        """Update the progress bar and record batch loss.

        Args:
            batch: Zero-based batch index.
            loss: Scalar loss for this batch.
            state: Current pipeline state.
        """
        if self._pbar is not None:
            self._pbar.update(1)
            self._pbar.set_postfix(loss=f"{loss:.4f}")
        self._losses.append(loss)

    def on_epoch_end(
        self, epoch: int, state: PipelineState
    ) -> None:
        """Close the progress bar for this epoch.

        Args:
            epoch: Zero-based epoch index.
            state: Current pipeline state.
        """
        if self._pbar is not None:
            self._pbar.close()
            self._pbar = None

        self._losses.clear()
