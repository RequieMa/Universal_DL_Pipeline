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
    Loss,
    LossProtocol,
    ModelProtocol,
    OptimizerProtocol,
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
        self._last_input = inputs
        x = inputs if is_torch else torch.as_tensor(inputs)
        # Match the module's weight dtype (e.g. float32 nn.Linear vs float64
        # numpy CSV data) so the matmul doesn't error on a dtype mismatch.
        if not is_torch:
            try:
                param_dtype = next(self._module.parameters()).dtype
            except StopIteration:
                param_dtype = x.dtype
            if x.dtype != param_dtype:
                x = x.to(dtype=param_dtype)
        output = self._module(x)
        if is_torch:
            return output
        # Detach so the numpy input→numpy output contract holds. For torch
        # training, TorchLoss re-runs the module on ``_last_input`` to
        # rebuild the autograd graph during backward().
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


class TorchLoss(LossProtocol):
    """Config-driven wrapper for ``torch.nn.*`` loss functions.

    Selects a torch loss class by name and forwards constructor kwargs.
    ``forward()`` wraps the torch loss tensor in a :class:`Loss` with a
    ``_backward_fn`` that calls ``result.backward()`` on the autograd
    graph.

    Usage::

        loss_fn = TorchLoss("CrossEntropyLoss", label_smoothing=0.1)
        loss = loss_fn(predictions, targets)
        loss.backward()  # calls tensor.backward() on the autograd graph

    Args:
        loss_name: Name of a ``torch.nn`` loss class
            (e.g., ``"CrossEntropyLoss"``, ``"MSELoss"``, ``"BCELoss"``).
        model: Optional :class:`TorchModel`. When provided, ``backward()``
            re-runs the module's forward on the cached inputs to rebuild the
            autograd graph and call ``tensor.backward()`` — needed because
            ``TorchModel.forward`` returns a detached numpy array for numpy
            input, which would otherwise sever the graph and make training
            impossible. When ``None``, ``backward()`` uses the graph built
            at ``forward()`` time (torch-tensor inputs).
        **kwargs: Forwarded to the torch loss constructor.
    """

    def __init__(
        self, loss_name: str, model: TorchModel | None = None, **kwargs: Any
    ) -> None:
        """Create a TorchLoss by name.

        Args:
            loss_name: ``torch.nn`` class name.
            model: Optional :class:`TorchModel` to rebuild the graph through
                during ``backward()`` (see class docstring).
            **kwargs: Arguments forwarded to the loss constructor.

        Raises:
            AttributeError: If ``loss_name`` is not found in ``torch.nn``.
        """
        import torch.nn as nn

        self._model = model
        loss_cls = getattr(nn, loss_name)
        self._loss = loss_cls(**kwargs)

    def forward(self, predictions: ArrayLike, targets: ArrayLike) -> Loss:
        """Compute loss.

        Converts numpy inputs to tensors via :func:`torch.as_tensor`
        (zero-copy when already tensors).

        Args:
            predictions: Model output.
            targets: Ground truth labels.

        Returns:
            :class:`Loss` with scalar value and ``_backward_fn``.
        """
        import torch

        p = torch.as_tensor(predictions)
        t = torch.as_tensor(targets)

        if self._model is not None:
            model = self._model
            loss_fn = self._loss

            def _backward() -> None:
                # Rebuild the graph: the numpy input→numpy output forward
                # detaches, so recompute through the module on the model's
                # cached input, then backprop to populate param gradients.
                raw_x = getattr(model, "_last_input", predictions)
                x = (
                    raw_x
                    if isinstance(raw_x, torch.Tensor)
                    else torch.as_tensor(raw_x)
                )
                param_dtype = next(model._module.parameters()).dtype
                if x.dtype != param_dtype:
                    x = x.to(dtype=param_dtype)
                re_pred = model._module(x)
                target_t = (
                    t if isinstance(t, torch.Tensor) else torch.as_tensor(t)
                )
                result = loss_fn(re_pred, target_t)
                result.backward()

            result = self._loss(p, t)
            return Loss(value=float(result.detach().cpu()), _backward_fn=_backward)

        result = self._loss(p, t)
        return Loss(value=float(result.detach().cpu()), _backward_fn=result.backward)


class TorchOptimizer(OptimizerProtocol):
    """Config-driven wrapper for ``torch.optim.*`` optimizers.

    Selects a torch optimizer class by name, forwards constructor kwargs,
    and delegates ``step()`` / ``zero_grad()``. Only :class:`Parameter`
    instances whose ``data`` is a ``torch.Tensor`` are passed to the
    underlying optimizer — non-tensor parameters (e.g., numpy arrays) are
    silently skipped.

    Usage::

        opt = TorchOptimizer(model.parameters(), "Adam", lr=0.001)
        opt.zero_grad()
        loss.backward()
        opt.step()

    Args:
        parameters: Iterable of :class:`Parameter` objects (live refs).
        optimizer_name: Name of a ``torch.optim`` class
            (e.g., ``"SGD"``, ``"Adam"``, ``"AdamW"``).
        **kwargs: Forwarded to the torch optimizer constructor.
    """

    def __init__(
        self,
        parameters: Iterable[Parameter],
        optimizer_name: str,
        **kwargs: Any,
    ) -> None:
        """Create a TorchOptimizer by name.

        Args:
            parameters: Live Parameter references (from ``model.parameters()``).
            optimizer_name: ``torch.optim`` class name.
            **kwargs: Arguments forwarded to the optimizer constructor
                (lr, momentum, weight_decay, etc.).

        Raises:
            AttributeError: If ``optimizer_name`` is not found in ``torch.optim``.
        """
        import torch
        import torch.optim as optim

        opt_cls = getattr(optim, optimizer_name)
        self._params = [p for p in parameters if isinstance(p.data, torch.Tensor)]
        self._opt = opt_cls([p.data for p in self._params], **kwargs)

    def step(self) -> None:
        """Update parameters using accumulated gradients.

        Delegates to ``self._opt.step()``.
        """
        self._opt.step()

    def zero_grad(self) -> None:
        """Reset all gradients to zero.

        Zeros the underlying data tensors in-place (via
        ``zero_grad(set_to_none=False)``) so gradients remain ``None``-free
        for inspection, and also zeros any :class:`Parameter` ``grad`` fields
        that hold their own tensor. Non-tensor parameters are skipped.

        Keeping gradients as zero tensors (rather than ``None``) matches the
        core :class:`Parameter` contract, where ``grad`` is a live, inspectable
        value.
        """
        import torch

        self._opt.zero_grad(set_to_none=False)
        for p in self._params:
            grad = p.grad
            if isinstance(grad, torch.Tensor):
                with torch.no_grad():
                    grad.zero_()
