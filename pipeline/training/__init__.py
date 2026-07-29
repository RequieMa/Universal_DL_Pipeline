"""Training loop orchestration.

Provides the :class:`TrainLoop` class that manages the per-epoch, per-batch
training lifecycle. Hook dispatch (epoch start/end, batch end) is integrated
so that cross-cutting concerns like logging, checkpointing, and early stopping
are transparent to the loop.

Usage::

    loop = TrainLoop(model, data_stream, optimizer, loss_fn, num_epochs, hooks)
    loop.run(state)
"""
from pipeline.training.train_loop import TrainLoop

__all__ = ["TrainLoop"]
