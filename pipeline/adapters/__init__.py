"""Framework adapters that satisfy pipeline protocols.

Each adapter wraps a specific framework (sklearn, numpy, PyTorch)
and satisfies :class:`ModelProtocol`, :class:`LossProtocol`, or
:class:`OptimizerProtocol` so the pipeline can use it without
knowing which framework is underneath.

Usage::

    from pipeline.adapters import SklearnModel, NumpyModel

Imports are lazy — sklearn is only imported when :class:`SklearnModel`
is instantiated, not when the module loads.
"""
from pipeline.adapters.sklearn_adapter import SklearnModel, StubLoss, StubOptimizer
from pipeline.adapters.numpy_adapter import NumpyModel, NumpyOptimizer

__all__ = [
    "SklearnModel",
    "StubLoss",
    "StubOptimizer",
    "NumpyModel",
    "NumpyOptimizer",
]
