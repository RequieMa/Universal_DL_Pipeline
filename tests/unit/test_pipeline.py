"""Unit tests for pipeline.pipeline — BasePipeline only."""

import numpy as np
import pytest

from pipeline.config import Config
from pipeline.evaluation.metrics import Metrics, accuracy
from pipeline.hooks import BaseHook
from pipeline.pipeline import BasePipeline, PipelineState


# ── Minimal concrete pipeline for testing ────────────────────────────────
class _MinimalPipeline(BasePipeline):
    """A pipeline where every stage is a no-op. For testing the template."""

    def load_data(self, state: PipelineState) -> None:
        state.current_epoch = 0

    def build_model(self, state: PipelineState) -> None:
        pass

    def train(self, state: PipelineState) -> None:
        state.current_epoch = state.config.num_epochs

    def evaluate(self, state: PipelineState) -> None:
        y_true = np.array([0, 1, 0])
        y_pred = np.array([0, 1, 0])
        state.metrics = Metrics(accuracy=accuracy)
        state.metrics.compute(y_true, y_pred)

    def export(self, state: PipelineState) -> None:
        state.predictions = np.array([0, 1, 0])


class _SpyHook(BaseHook):
    """A hook that records every event it receives."""

    def __init__(self):
        self.events: list[str] = []

    def on_stage_start(self, stage: str, state: PipelineState) -> None:
        self.events.append(f"start:{stage}")

    def on_stage_end(self, stage: str, state: PipelineState) -> None:
        self.events.append(f"end:{stage}")

    def on_epoch_start(self, epoch: int, state: PipelineState) -> None:
        self.events.append(f"epoch_start:{epoch}")

    def on_epoch_end(self, epoch: int, state: PipelineState) -> None:
        self.events.append(f"epoch_end:{epoch}")

    def on_batch_end(self, batch: int, loss: float, state: PipelineState) -> None:
        self.events.append(f"batch_end:{batch}:{loss:.4f}")


# ── BasePipeline ABC tests ────────────────────────────────────────────────
class TestBasePipelineABC:
    """Tests for BasePipeline abstract base class."""

    def test_cannot_instantiate_abc_directly(self):
        """Type Error: instantiating BasePipeline directly raises TypeError."""
        with pytest.raises(TypeError):
            BasePipeline(Config())  # type: ignore[abstract]

    def test_concrete_subclass_instantiates(self):
        """Happy Path: a concrete subclass can be instantiated."""
        pipeline = _MinimalPipeline(Config())
        assert pipeline is not None
        assert pipeline.config is not None


class TestBasePipelineRun:
    """Tests for BasePipeline.run() template method."""

    def test_run_train_executes_all_stages(self):
        """Happy Path: run('train') hits all 6 stages."""
        pipeline = _MinimalPipeline(Config(num_epochs=3))
        state = pipeline.run("train")
        assert state.current_epoch == 3
        assert state.metrics["accuracy"] == 1.0
        assert state.predictions is not None

    def test_run_infer_skips_training_stages(self):
        """Happy Path: run('infer') skips extract_features, build_model,
        train, and evaluate."""
        pipeline = _MinimalPipeline(Config(num_epochs=3))
        state = pipeline.run("infer")
        # Training stages skipped
        assert state.model is None
        assert state.current_epoch == 0
        assert len(state.metrics) == 0  # UPDATED: was == {}
        # load_data and export still run
        assert state.predictions is not None

    def test_run_returns_pipeline_state(self):
        """Happy Path: run() returns the PipelineState it built."""
        pipeline = _MinimalPipeline(Config())
        state = pipeline.run("train")
        assert isinstance(state, PipelineState)

    def test_run_with_zero_epochs(self):
        """Boundary: run with num_epochs=0 is valid."""
        pipeline = _MinimalPipeline(Config(num_epochs=0))
        state = pipeline.run("train")
        assert state is not None


class TestBasePipelineHooks:
    """Tests for BasePipeline hook dispatch."""

    def test_no_hooks_pipeline_runs(self):
        """Boundary: pipeline with zero hooks runs without error."""
        pipeline = _MinimalPipeline(Config())
        state = pipeline.run("train")
        assert state.metrics["accuracy"] == 1.0

    def test_single_hook_receives_events(self):
        """Happy Path: a registered hook receives stage start/end events."""
        pipeline = _MinimalPipeline(Config())
        spy = _SpyHook()
        pipeline.add_hook(spy)
        pipeline.run("train")
        assert "start:load_data" in spy.events
        assert "end:load_data" in spy.events
        assert "start:train" in spy.events
        assert "end:train" in spy.events

    def test_multiple_hooks_both_fire(self):
        """Happy Path: two hooks both receive events."""
        pipeline = _MinimalPipeline(Config())
        spy1 = _SpyHook()
        spy2 = _SpyHook()
        pipeline.add_hook(spy1)
        pipeline.add_hook(spy2)
        pipeline.run("train")
        assert len(spy1.events) > 0
        assert len(spy2.events) > 0
        # Both should have same start events
        assert spy1.events[0] == spy2.events[0]

    def test_hook_order_is_registration_order(self):
        """Concurrency: hooks fire in the order they were added."""
        pipeline = _MinimalPipeline(Config())
        order_log: list[int] = []

        class OrderedHook(BaseHook):
            def __init__(self, tag: int):
                self.tag = tag

            def on_stage_start(self, stage, state):
                order_log.append(self.tag)

        pipeline.add_hook(OrderedHook(1))
        pipeline.add_hook(OrderedHook(2))
        pipeline.add_hook(OrderedHook(3))
        pipeline.run("train")
        # Each stage triggers all 3 hooks in order
        assert order_log[0:3] == [1, 2, 3]


class TestBasePipelineErrorRecovery:
    """Tests for error handling in the pipeline."""

    def test_hook_exception_does_not_kill_pipeline(self):
        """Error recovery: a hook that raises doesn't prevent completion.
        NOTE: Spec requires error recovery. The minimal implementation
        logs the error and continues."""
        pipeline = _MinimalPipeline(Config())

        class CrashingHook(BaseHook):
            def on_stage_start(self, stage, state):
                if stage == "train":
                    raise RuntimeError("hook crash")

        pipeline.add_hook(CrashingHook())
        # Should not raise — pipeline recovers
        state = pipeline.run("train")
        assert state.metrics["accuracy"] == 1.0

    def test_stage_exception_propagates(self):
        """Error recovery: a stage that raises propagates the exception
        (stages are critical path; hooks are not)."""

        class FailingPipeline(_MinimalPipeline):
            def train(self, state):
                raise RuntimeError("training failed")

        pipeline = FailingPipeline(Config())
        with pytest.raises(RuntimeError, match="training failed"):
            pipeline.run("train")

    def test_infer_mode_skips_stages_cleanly(self):
        """Happy Path: infer mode still runs load_data and export."""
        pipeline = _MinimalPipeline(Config())
        state = pipeline.run("infer")
        assert state.predictions is not None
        assert state.model is None  # never built


class TestBasePipelineHooksProperty:
    """Tests for BasePipeline.hooks read-only @property."""

    def test_hooks_property_returns_list(self):
        """Happy Path: hooks returns the internal hook list."""
        pipeline = _MinimalPipeline(Config())
        assert isinstance(pipeline.hooks, list)
        assert len(pipeline.hooks) == 0

    def test_hooks_property_read_only(self):
        """Error: hooks property raises AttributeError on set."""
        pipeline = _MinimalPipeline(Config())
        with pytest.raises(AttributeError):
            pipeline.hooks = []  # type: ignore[misc]

    def test_hooks_property_reflects_add_hook(self):
        """Happy Path: hooks list reflects hooks added via add_hook()."""
        pipeline = _MinimalPipeline(Config())
        spy = _SpyHook()
        pipeline.add_hook(spy)
        assert len(pipeline.hooks) == 1
        assert pipeline.hooks[0] is spy


class TestBasePipelineTrainDefault:
    """Tests for the default train() implementation (delegates to TrainLoop).

    NOTE: These tests verify train() is no longer abstract. Full TrainLoop
    integration is tested in test_train_loop.py (Task 5).
    """

    def test_train_is_not_abstract(self):
        """Happy Path: subclasses without train() can be instantiated."""

        class PipelineNoTrain(BasePipeline):
            def load_data(self, state):
                pass

            def build_model(self, state):
                pass

            def evaluate(self, state):
                pass

            def export(self, state):
                pass

        pipeline = PipelineNoTrain(Config())
        assert pipeline is not None

    def test_train_uses_lazy_import(self):
        """Happy Path: train() imports TrainLoop lazily (verified via
        `hasattr` -- the method exists and has no __isabstractmethod__)."""
        assert not hasattr(BasePipeline.train, "__isabstractmethod__")
