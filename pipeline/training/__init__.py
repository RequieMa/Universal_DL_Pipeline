"""Training loop orchestration and optimization components.

Provides the :class:`TrainLoop` class that manages the per-epoch,
per-batch training lifecycle, plus pure-numpy optimizer rules
(:class:`SGD`, :class:`Adam`) and loss functions (:class:`MSELoss`,
:class:`CrossEntropyLoss`).

Usage::

    from pipeline.training import TrainLoop, SGD, Adam, MSELoss, CrossEntropyLoss
"""

from pipeline.training.losses import CrossEntropyLoss, MSELoss
from pipeline.training.optimizers import SGD, Adam
from pipeline.training.train_loop import TrainLoop

__all__ = ["TrainLoop", "SGD", "Adam", "MSELoss", "CrossEntropyLoss"]
