"""PyTorch adapter — TorchModel, TorchLoss, TorchOptimizer.

Wraps torch.nn.Module, torch loss functions, and torch optimizers
to satisfy the pipeline protocols. All torch imports are lazy
(inside functions) so the core pipeline imports without PyTorch.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from pipeline.protocols import (
    ArrayLike,
    ModelProtocol,
    Parameter,
)


class TorchModel(ModelProtocol):
    """Wraps a :class:`torch.nn.Module` as a :class:`ModelProtocol`.

    ``forward()`` preserves the input format: torch tensors go straight
    through (zero-copy), numpy arrays are converted to/from torch.

    ``parameters()`` returns live :class:`Parameter` references to the
    underlying ``nn.Parameter`` tensors — the optimizer mutates them
    in-place, so gradient updates automatically flow back to the module.

    Usage::

        import torch.nn as nn
        module = nn.Sequential(nn.Linear(784, 128), nn.ReLU(), nn.Linear(128, 10))
        model = TorchModel(module)

    Args:
        module: Any ``torch.nn.Module`` instance.
    """

    def __init__(self, module: Any) -> None:
        """Wrap a torch.nn.Module.

        Args:
            module: A ``torch.nn.Module`` instance.
        """
        self._module = module

    # ------------------------------------------------------------------
    # ModelProtocol
    # ------------------------------------------------------------------

    def forward(self, inputs: ArrayLike) -> ArrayLike:
        """Run a forward pass.

        Format-preserving: ``torch.Tensor`` in → ``torch.Tensor`` out
        (zero-copy). ``numpy.ndarray`` in → ``numpy.ndarray`` out.

        Args:
            inputs: Input tensor/array.

        Returns:
            Model output in the same format as the input.
        """
        import torch  # type: ignore[import-not-found]

        is_torch = isinstance(inputs, torch.Tensor)
        x = inputs if is_torch else torch.as_tensor(inputs)
        output = self._module(x)
        if is_torch:
            return output
        return output.detach().cpu().numpy()

    def parameters(self) -> Iterable[Parameter]:
        """Yield live :class:`Parameter` references to nn.Parameter tensors.

        The returned ``Parameter.data`` and ``Parameter.grad`` point to
        the actual underlying torch tensors — mutations by the optimizer
        are reflected in the module.
        """
        for name, param in self._module.named_parameters():
            yield Parameter(
                data=param,
                grad=param.grad,
                name=name,
            )

    def train_mode(self) -> None:
        """Switch module to training mode (``module.train()``)."""
        self._module.train()

    def eval_mode(self) -> None:
        """Switch module to evaluation mode (``module.eval()``)."""
        self._module.eval()
