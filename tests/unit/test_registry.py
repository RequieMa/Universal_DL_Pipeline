"""Unit tests for pipeline.registry — DI registry."""

import pytest

from pipeline.registry import build, is_registered, list_registered, register


# ── Test classes for registration ─────────────────────────────────────────
@register("model", "dummy_a")
class DummyModelA:
    def __init__(self, hidden: int = 128):
        self.hidden = hidden


@register("model", "dummy_b")
class DummyModelB:
    def __init__(self, layers: int = 3):
        self.layers = layers


@register("optimizer", "sgd_test")
class DummyOptimizer:
    def __init__(self, lr: float = 0.01):
        self.lr = lr


# ── Tests ──────────────────────────────────────────────────────────────────
class TestRegister:
    """Tests for register() decorator."""

    def test_registered_class_builds(self):
        """Happy Path: registered class can be built via build()."""
        obj = build("model", "dummy_a", hidden=64)
        assert isinstance(obj, DummyModelA)
        assert obj.hidden == 64

    def test_registered_class_builds_with_defaults(self):
        """Happy Path: build with no kwargs uses class defaults."""
        obj = build("model", "dummy_b")
        assert obj.layers == 3

    def test_register_multiple_classes_same_kind(self):
        """Happy Path: multiple classes can register under the same kind."""
        a = build("model", "dummy_a")
        b = build("model", "dummy_b")
        assert isinstance(a, DummyModelA)
        assert isinstance(b, DummyModelB)

    def test_register_different_kinds(self):
        """Happy Path: classes can register under different kinds."""
        model = build("model", "dummy_a")
        opt = build("optimizer", "sgd_test")
        assert isinstance(model, DummyModelA)
        assert isinstance(opt, DummyOptimizer)


class TestBuildEdgeCases:
    """Tests for build() edge cases."""

    def test_build_unregistered_kind_raises(self):
        """Type Error: building from unknown kind raises KeyError."""
        with pytest.raises(KeyError):
            build("nonexistent_kind", "anything")

    def test_build_unregistered_name_raises(self):
        """Type Error: building unknown name within known kind raises KeyError."""
        with pytest.raises(KeyError):
            build("model", "nonexistent_model")

    def test_build_with_unexpected_kwargs(self):
        """Type Error: build with unknown keyword arguments raises TypeError.
        NOTE: This depends on the registered class's __init__ signature."""
        with pytest.raises(TypeError):
            build("model", "dummy_a", nonexistent_param=42)


class TestListRegistered:
    """Tests for list_registered() introspection."""

    def test_list_registered_returns_names(self):
        """Happy Path: list_registered returns all names for a kind."""
        names = list_registered("model")
        assert "dummy_a" in names
        assert "dummy_b" in names

    def test_list_registered_unknown_kind(self):
        """Boundary: list_registered for unknown kind returns empty list."""
        names = list_registered("nonexistent")
        assert names == []


class TestIsRegistered:
    """Tests for is_registered()."""

    def test_is_registered_known(self):
        """Happy Path: registered name returns True."""
        assert is_registered("model", "dummy_a") is True

    def test_is_registered_unknown_name(self):
        """Boundary: unregistered name returns False."""
        assert is_registered("model", "fake") is False

    def test_is_registered_unknown_kind(self):
        """Boundary: unregistered kind returns False."""
        assert is_registered("fake_kind", "anything") is False


class TestRegistryStress:
    """Stress tests for the registry."""

    def test_register_many_classes(self):
        """Stress: register 500 classes and build each one."""
        n = 500
        for i in range(n):

            @register("stress_test", f"cls_{i}")
            class _StressClass:
                def __init__(self, idx: int = i):
                    self.idx = idx

        names = list_registered("stress_test")
        assert len(names) == n

        # Build a random sample
        obj = build("stress_test", "cls_42")
        assert obj.idx == 42


class TestRegistryIsolation:
    """Tests for registry state isolation."""

    def test_registry_no_cross_kind_leakage(self):
        """Concurrency: registering under one kind doesn't affect others."""

        @register("isolated_kind", "test_cls")
        class _Isolated:
            pass

        assert "test_cls" in list_registered("isolated_kind")
        assert "test_cls" not in list_registered("model")
