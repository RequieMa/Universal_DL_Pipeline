"""Dependency injection registry.

A minimal type-based registry: one dict, two functions, zero magic.
Classes are registered by ``(kind, name)`` pairs and built by the
same keys with keyword arguments forwarded to ``__init__``.

Usage::

    from pipeline.registry import register, build

    @register("backbone", "resnet18")
    class ResNet18:
        def __init__(self, num_classes: int = 10):
            ...

    model = build("backbone", "resnet18", num_classes=100)

NOTE: The registry is a module-level ``dict``. Registration happens
at import time when the decorator runs. This is intentional — it keeps
the system simple enough for a beginner to understand in one reading.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

# ── Internal storage ───────────────────────────────────────────────────
# WHY: A flat dict of dicts. _REGISTRY["backbone"]["resnet18"] = ResNet18
# No metaclasses, no YAML scanning, no dynamic imports. A student can
# print(_REGISTRY) and see everything registered.
_REGISTRY: dict[str, dict[str, type]] = {}


# ── Public API ────────────────────────────────────────────────────────
def register(kind: str, name: str) -> Callable[[type], type]:
    """Decorator: register a class under ``(kind, name)``.

    Args:
        kind: Component category (e.g., ``"backbone"``, ``"optimizer"``).
        name: Unique name within the kind (e.g., ``"resnet18"``).

    Returns:
        A decorator that registers the class and returns it unchanged.

    Usage::

        @register("backbone", "resnet18")
        class ResNet18:
            ...
    """

    def decorator(cls: type) -> type:
        _REGISTRY.setdefault(kind, {})[name] = cls
        return cls

    return decorator


def build(kind: str, name: str, **kwargs: Any) -> Any:
    """Build a registered component by ``(kind, name)``.

    Args:
        kind: Component category.
        name: Registered name within the category.
        **kwargs: Forwarded to the class ``__init__``.

    Returns:
        An instance of the registered class.

    Raises:
        KeyError: If ``(kind, name)`` is not registered.
    """
    try:
        cls = _REGISTRY[kind][name]
    except KeyError:
        raise KeyError(
            f"No component registered under kind={kind!r}, name={name!r}. "
            f"Available names for {kind!r}: {list_registered(kind)}"
        ) from None
    return cls(**kwargs)


def list_registered(kind: str) -> list[str]:
    """Return all registered names for a given kind.

    Useful for debugging and introspection::

        >>> list_registered("backbone")
        ["resnet18", "simple_cnn", "mlp"]

    Args:
        kind: Component category.

    Returns:
        List of registered names (empty if kind is unknown).
    """
    return list(_REGISTRY.get(kind, {}).keys())


def is_registered(kind: str, name: str) -> bool:
    """Check whether ``(kind, name)`` is registered.

    Args:
        kind: Component category.
        name: Component name.

    Returns:
        ``True`` if the component is registered.
    """
    return name in _REGISTRY.get(kind, {})
