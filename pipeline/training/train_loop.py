"""Training lifecycle manager with hook dispatch.

See :class:`TrainLoop` for details.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable

    from pipeline.hooks import BaseHook
    from pipeline.pipeline import PipelineState
    from pipeline.protocols import DataStream, LossProtocol, ModelProtocol, OptimizerProtocol

_logger = logging.getLogger(__name__)


class TrainLoop:
    """Training lifecycle manager.

    Runs the per-epoch, per-batch training loop and dispatches hook events
    (``on_epoch_start``, ``on_epoch_end``, ``on_batch_end``) so that
    cross-cutting concerns like logging and early stopping are transparent.

    Usage::

        loop = TrainLoop(model, data_stream, optimizer, loss_fn, num_epochs, hooks)
        loop.run(state)

    The loop updates ``state.history["loss"]`` (one float per batch per epoch),
    ``state.current_epoch``, and respects ``state.should_stop`` for early stopping.
    """

    def __init__(
        self,
        model: ModelProtocol,
        data_stream: DataStream,
        optimizer: OptimizerProtocol,
        loss_fn: LossProtocol,
        num_epochs: int,
        hooks: Iterable[BaseHook] | None = None,
    ) -> None:
        """Store loop configuration.

        Args:
            model: A model satisfying :class:`ModelProtocol`.
            data_stream: Iterable producing :class:`Batch` objects.
            optimizer: Optimizer for gradient-based parameter updates.
            loss_fn: Loss function (callable or with ``forward``).
            num_epochs: Number of full passes over the data stream.
            hooks: Optional iterable of hook objects. Defaults to empty.
        """
        self._model = model
        self._data_stream = data_stream
        self._optimizer = optimizer
        self._loss_fn = loss_fn
        self._num_epochs = num_epochs
        self._hooks: list[BaseHook] = list(hooks) if hooks is not None else []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, state: PipelineState) -> None:
        """Execute the training loop.

        For each epoch, iterates over all batches, runs the forward-loss-
        backward-step cycle, and dispatches hook events. Populates
        ``state.history["loss"]`` with per-batch loss values.

        Args:
            state: Shared pipeline state. The loop reads/writes
                ``state.history``, ``state.current_epoch``, and
                ``state.should_stop``.
        """
        state.history = {"loss": []}
        self._model.train_mode()

        for epoch in range(self._num_epochs):
            if state.should_stop:
                break
            state.current_epoch = epoch
            self._notify("on_epoch_start", epoch, state)

            for batch_idx, batch in enumerate(self._data_stream):
                predictions = self._model.forward(batch.inputs)
                loss = self._loss_fn(predictions, batch.targets)
                loss_value = float(loss)

                self._optimizer.zero_grad()
                if hasattr(loss, "backward"):
                    loss.backward()
                self._optimizer.step()

                state.history["loss"].append(loss_value)
                self._notify("on_batch_end", batch_idx, loss_value, state)

                if state.should_stop:
                    break

            self._notify("on_epoch_end", epoch, state)

            state.current_epoch = epoch + 1

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _notify(self, event: str, *args: object) -> None:
        """Dispatch ``event`` to all registered hooks.

        Hook exceptions are caught and logged -- they do not interrupt
        the training loop.

        Args:
            event: Hook method name (e.g., ``"on_batch_end"``).
            *args: Arguments forwarded to the hook method.
        """
        for hook in self._hooks:
            try:
                getattr(hook, event)(*args)
            except Exception:
                _logger.exception(
                    "Hook %s.%s raised an exception (ignored)",
                    hook.__class__.__name__,
                    event,
                )
