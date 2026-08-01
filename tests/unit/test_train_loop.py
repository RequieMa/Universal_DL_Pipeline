"""Unit tests for TrainLoop — the training lifecycle manager with hook dispatch."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from pipeline.hooks import BaseHook
from pipeline.pipeline import PipelineState
from pipeline.protocols import Batch, Loss

if TYPE_CHECKING:
    from pipeline.config import Config

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_state(config: Config | None = None) -> PipelineState:
    """Build a minimal PipelineState for TrainLoop tests."""
    from pipeline.config import Config

    return PipelineState(config=config or Config(num_epochs=2))


# ---------------------------------------------------------------------------
# Import test (TDD: module doesn't exist yet)
# ---------------------------------------------------------------------------


class TestTrainLoopImport:
    """TrainLoop lives in pipeline.training.train_loop."""

    def test_module_exists(self):
        """TDD step 1: module not found until created."""
        from pipeline.training.train_loop import TrainLoop  # noqa: F811

        assert TrainLoop is not None


# ---------------------------------------------------------------------------
# Construction tests
# ---------------------------------------------------------------------------


class TestTrainLoopConstruction:
    """TrainLoop can be instantiated from model, data, optimizer, loss, epochs, hooks."""

    def test_constructs_with_all_args(self):
        """Happy Path: all required args produce a valid TrainLoop."""
        from pipeline.training.train_loop import TrainLoop
        from tests.unit.conftest import FakeDataStream, FakeLoss, FakeModel, FakeOptimizer

        stream = FakeDataStream([Batch(inputs=np.array([[1.0]]), targets=np.array([[2.0]]))])
        loop = TrainLoop(
            model=FakeModel(),
            data_stream=stream,
            optimizer=FakeOptimizer(),
            loss_fn=FakeLoss(),
            num_epochs=3,
            hooks=[],
        )
        assert loop is not None

    def test_constructs_without_hooks(self):
        """Boundary: hooks list is empty by default."""
        from pipeline.training.train_loop import TrainLoop
        from tests.unit.conftest import FakeDataStream, FakeLoss, FakeModel, FakeOptimizer

        stream = FakeDataStream([Batch(inputs=np.array([[1.0]]), targets=np.array([[2.0]]))])
        loop = TrainLoop(
            model=FakeModel(),
            data_stream=stream,
            optimizer=FakeOptimizer(),
            loss_fn=FakeLoss(),
            num_epochs=3,
        )
        assert loop is not None

    def test_constructs_with_single_batch(self):
        """Boundary: a single batch is valid."""
        from pipeline.training.train_loop import TrainLoop
        from tests.unit.conftest import FakeDataStream, FakeLoss, FakeModel, FakeOptimizer

        stream = FakeDataStream([Batch(inputs=np.array([[1.0]]), targets=np.array([[2.0]]))])
        loop = TrainLoop(
            model=FakeModel(),
            data_stream=stream,
            optimizer=FakeOptimizer(),
            loss_fn=FakeLoss(),
            num_epochs=1,
        )
        assert loop is not None

    def test_constructs_with_zero_epochs(self):
        """Boundary: zero epochs is valid (no training iterations)."""
        from pipeline.training.train_loop import TrainLoop
        from tests.unit.conftest import FakeDataStream, FakeLoss, FakeModel, FakeOptimizer

        stream = FakeDataStream([Batch(inputs=np.array([[1.0]]), targets=np.array([[2.0]]))])
        loop = TrainLoop(
            model=FakeModel(),
            data_stream=stream,
            optimizer=FakeOptimizer(),
            loss_fn=FakeLoss(),
            num_epochs=0,
        )
        assert loop is not None


# ---------------------------------------------------------------------------
# run() output tests
# ---------------------------------------------------------------------------


class TestTrainLoopRunHistory:
    """TrainLoop.run() populates state.history correctly."""

    def test_history_contains_loss(self):
        """Happy Path: run() writes state.history['loss'] as list of floats."""
        from pipeline.training.train_loop import TrainLoop
        from tests.unit.conftest import FakeDataStream, FakeLoss, FakeModel, FakeOptimizer

        state = _make_state()
        stream = FakeDataStream(
            [
                Batch(inputs=np.array([[1.0], [2.0]]), targets=np.array([[2.0], [4.0]])),
            ]
        )
        loop = TrainLoop(
            model=FakeModel(),
            data_stream=stream,
            optimizer=FakeOptimizer(),
            loss_fn=FakeLoss(),
            num_epochs=2,
            hooks=[],
        )
        loop.run(state)
        assert "loss" in state.history
        assert isinstance(state.history["loss"], list)
        # 2 epochs x 1 batch = 2 entries
        assert len(state.history["loss"]) == 2
        assert all(isinstance(v, float) for v in state.history["loss"])

    def test_history_length_equals_epochs_times_batches(self):
        """Happy Path: loss history has one entry per batch per epoch."""
        from pipeline.training.train_loop import TrainLoop
        from tests.unit.conftest import FakeDataStream, FakeLoss, FakeModel, FakeOptimizer

        state = _make_state()
        stream = FakeDataStream(
            [
                Batch(inputs=np.array([[1.0]]), targets=np.array([[2.0]])),
                Batch(inputs=np.array([[3.0]]), targets=np.array([[6.0]])),
                Batch(inputs=np.array([[5.0]]), targets=np.array([[10.0]])),
            ]
        )
        loop = TrainLoop(
            model=FakeModel(),
            data_stream=stream,
            optimizer=FakeOptimizer(),
            loss_fn=FakeLoss(),
            num_epochs=4,
            hooks=[],
        )
        loop.run(state)
        assert len(state.history["loss"]) == 4 * 3  # 12 entries

    def test_zero_epochs_produces_empty_history(self):
        """Boundary: zero epochs results in empty loss history."""
        from pipeline.training.train_loop import TrainLoop
        from tests.unit.conftest import FakeDataStream, FakeLoss, FakeModel, FakeOptimizer

        state = _make_state()
        stream = FakeDataStream(
            [
                Batch(inputs=np.array([[1.0]]), targets=np.array([[2.0]])),
            ]
        )
        loop = TrainLoop(
            model=FakeModel(),
            data_stream=stream,
            optimizer=FakeOptimizer(),
            loss_fn=FakeLoss(),
            num_epochs=0,
            hooks=[],
        )
        loop.run(state)
        assert len(state.history["loss"]) == 0

    def test_larger_loss_for_mismatched_prediction(self):
        """Happy Path: worse predictions produce larger loss values."""
        from pipeline.training.train_loop import TrainLoop
        from tests.unit.conftest import FakeDataStream, FakeLoss, FakeModel, FakeOptimizer

        state = _make_state()
        stream = FakeDataStream(
            [
                Batch(inputs=np.array([[100.0]]), targets=np.array([[0.0]])),
            ]
        )
        loop = TrainLoop(
            model=FakeModel(),
            data_stream=stream,
            optimizer=FakeOptimizer(),
            loss_fn=FakeLoss(),
            num_epochs=1,
            hooks=[],
        )
        loop.run(state)
        # FakeModel: x*2, FakeLoss: MSE, target=0, pred=200 -> (200-0)^2 = 40000
        assert state.history["loss"][0] == pytest.approx(40000.0)


# ---------------------------------------------------------------------------
# run() state tests
# ---------------------------------------------------------------------------


class TestTrainLoopStateUpdates:
    """TrainLoop.run() updates state.current_epoch and should_stop."""

    def test_current_epoch_before_training(self):
        """Boundary: current_epoch is 0 before run() is called."""
        from pipeline.training.train_loop import TrainLoop
        from tests.unit.conftest import FakeDataStream, FakeLoss, FakeModel, FakeOptimizer

        state = _make_state()
        stream = FakeDataStream(
            [
                Batch(inputs=np.array([[1.0]]), targets=np.array([[2.0]])),
            ]
        )
        _ = TrainLoop(
            model=FakeModel(),
            data_stream=stream,
            optimizer=FakeOptimizer(),
            loss_fn=FakeLoss(),
            num_epochs=3,
            hooks=[],
        )
        assert state.current_epoch == 0

    def test_current_epoch_after_run(self):
        """Happy Path: current_epoch equals num_epochs after run()."""
        from pipeline.training.train_loop import TrainLoop
        from tests.unit.conftest import FakeDataStream, FakeLoss, FakeModel, FakeOptimizer

        state = _make_state()
        stream = FakeDataStream(
            [
                Batch(inputs=np.array([[1.0]]), targets=np.array([[2.0]])),
            ]
        )
        loop = TrainLoop(
            model=FakeModel(),
            data_stream=stream,
            optimizer=FakeOptimizer(),
            loss_fn=FakeLoss(),
            num_epochs=5,
            hooks=[],
        )
        loop.run(state)
        assert state.current_epoch == 5

    def test_history_is_dict_with_loss_key(self):
        """Happy Path: history is a dict with at least 'loss' key."""
        from pipeline.training.train_loop import TrainLoop
        from tests.unit.conftest import FakeDataStream, FakeLoss, FakeModel, FakeOptimizer

        state = _make_state()
        stream = FakeDataStream(
            [
                Batch(inputs=np.array([[1.0]]), targets=np.array([[2.0]])),
            ]
        )
        loop = TrainLoop(
            model=FakeModel(),
            data_stream=stream,
            optimizer=FakeOptimizer(),
            loss_fn=FakeLoss(),
            num_epochs=1,
            hooks=[],
        )
        loop.run(state)
        assert isinstance(state.history, dict)
        assert "loss" in state.history


# ---------------------------------------------------------------------------
# Hook dispatch tests
# ---------------------------------------------------------------------------


class _SpyHook(BaseHook):
    """Hook that records all training events."""

    def __init__(self) -> None:
        self.events: list[str] = []

    def on_epoch_start(self, epoch: int, state: PipelineState) -> None:
        self.events.append(f"epoch_start:{epoch}")

    def on_epoch_end(self, epoch: int, state: PipelineState) -> None:
        self.events.append(f"epoch_end:{epoch}")

    def on_batch_end(self, batch: int, loss: float, state: PipelineState) -> None:
        self.events.append(f"batch_end:{batch}:{loss:.4f}")


class TestTrainLoopHookDispatch:
    """TrainLoop dispatches on_epoch_start, on_epoch_end, on_batch_end to hooks."""

    def test_on_epoch_start_fires_per_epoch(self):
        """Happy Path: on_epoch_start fires once per epoch."""
        from pipeline.training.train_loop import TrainLoop
        from tests.unit.conftest import FakeDataStream, FakeLoss, FakeModel, FakeOptimizer

        state = _make_state()
        stream = FakeDataStream(
            [
                Batch(inputs=np.array([[1.0]]), targets=np.array([[2.0]])),
            ]
        )
        spy = _SpyHook()
        loop = TrainLoop(
            model=FakeModel(),
            data_stream=stream,
            optimizer=FakeOptimizer(),
            loss_fn=FakeLoss(),
            num_epochs=3,
            hooks=[spy],
        )
        loop.run(state)
        epoch_starts = [e for e in spy.events if e.startswith("epoch_start")]
        assert len(epoch_starts) == 3

    def test_on_epoch_end_fires_per_epoch(self):
        """Happy Path: on_epoch_end fires once per epoch."""
        from pipeline.training.train_loop import TrainLoop
        from tests.unit.conftest import FakeDataStream, FakeLoss, FakeModel, FakeOptimizer

        state = _make_state()
        stream = FakeDataStream(
            [
                Batch(inputs=np.array([[1.0]]), targets=np.array([[2.0]])),
            ]
        )
        spy = _SpyHook()
        loop = TrainLoop(
            model=FakeModel(),
            data_stream=stream,
            optimizer=FakeOptimizer(),
            loss_fn=FakeLoss(),
            num_epochs=3,
            hooks=[spy],
        )
        loop.run(state)
        epoch_ends = [e for e in spy.events if e.startswith("epoch_end")]
        assert len(epoch_ends) == 3

    def test_on_batch_end_fires_per_batch(self):
        """Happy Path: on_batch_end fires for each batch in each epoch."""
        from pipeline.training.train_loop import TrainLoop
        from tests.unit.conftest import FakeDataStream, FakeLoss, FakeModel, FakeOptimizer

        state = _make_state()
        stream = FakeDataStream(
            [
                Batch(inputs=np.array([[1.0]]), targets=np.array([[2.0]])),
                Batch(inputs=np.array([[3.0]]), targets=np.array([[6.0]])),
            ]
        )
        spy = _SpyHook()
        loop = TrainLoop(
            model=FakeModel(),
            data_stream=stream,
            optimizer=FakeOptimizer(),
            loss_fn=FakeLoss(),
            num_epochs=4,
            hooks=[spy],
        )
        loop.run(state)
        batch_ends = [e for e in spy.events if e.startswith("batch_end")]
        assert len(batch_ends) == 4 * 2  # 4 epochs x 2 batches

    def test_batch_end_receives_loss_value(self):
        """Happy Path: on_batch_end receives the actual loss as a float."""
        from pipeline.training.train_loop import TrainLoop
        from tests.unit.conftest import FakeDataStream, FakeLoss, FakeModel, FakeOptimizer

        state = _make_state()
        stream = FakeDataStream(
            [
                Batch(inputs=np.array([[1.0]]), targets=np.array([[2.0]])),
            ]
        )
        spy = _SpyHook()
        loop = TrainLoop(
            model=FakeModel(),
            data_stream=stream,
            optimizer=FakeOptimizer(),
            loss_fn=FakeLoss(),
            num_epochs=1,
            hooks=[spy],
        )
        loop.run(state)
        # Spy event order: epoch_start(0), batch_end(0, 0.0), epoch_end(0)
        # batch_end is at index 1
        batch_event = spy.events[1]
        assert "batch_end:0:" in batch_event

    def test_multiple_hooks_all_fire(self):
        """Happy Path: two hooks both receive all events."""
        from pipeline.training.train_loop import TrainLoop
        from tests.unit.conftest import FakeDataStream, FakeLoss, FakeModel, FakeOptimizer

        state = _make_state()
        stream = FakeDataStream(
            [
                Batch(inputs=np.array([[1.0]]), targets=np.array([[2.0]])),
            ]
        )
        spy1 = _SpyHook()
        spy2 = _SpyHook()
        loop = TrainLoop(
            model=FakeModel(),
            data_stream=stream,
            optimizer=FakeOptimizer(),
            loss_fn=FakeLoss(),
            num_epochs=2,
            hooks=[spy1, spy2],
        )
        loop.run(state)
        assert len(spy1.events) == len(spy2.events)
        assert spy1.events[0] == spy2.events[0]


# ---------------------------------------------------------------------------
# Early stopping tests
# ---------------------------------------------------------------------------


class _StopHook(BaseHook):
    """Hook that sets should_stop after N epochs."""

    def __init__(self, stop_after: int) -> None:
        self.stop_after = stop_after

    def on_epoch_end(self, epoch: int, state: PipelineState) -> None:
        if epoch >= self.stop_after - 1:
            state.should_stop = True


class TestTrainLoopEarlyStopping:
    """TrainLoop respects state.should_stop from hooks."""

    def test_stops_after_first_epoch(self):
        """Happy Path: should_stop=True after epoch 0 stops after 1 epoch."""
        from pipeline.training.train_loop import TrainLoop
        from tests.unit.conftest import FakeDataStream, FakeLoss, FakeModel, FakeOptimizer

        state = _make_state()
        stream = FakeDataStream(
            [
                Batch(inputs=np.array([[1.0]]), targets=np.array([[2.0]])),
            ]
        )
        loop = TrainLoop(
            model=FakeModel(),
            data_stream=stream,
            optimizer=FakeOptimizer(),
            loss_fn=FakeLoss(),
            num_epochs=10,
            hooks=[_StopHook(stop_after=1)],
        )
        loop.run(state)
        assert state.current_epoch == 1  # stopped after epoch 1

    def test_stops_after_third_epoch(self):
        """Happy Path: should_stop after epoch 2 stops after 3 epochs."""
        from pipeline.training.train_loop import TrainLoop
        from tests.unit.conftest import FakeDataStream, FakeLoss, FakeModel, FakeOptimizer

        state = _make_state()
        stream = FakeDataStream(
            [
                Batch(inputs=np.array([[1.0]]), targets=np.array([[2.0]])),
            ]
        )
        loop = TrainLoop(
            model=FakeModel(),
            data_stream=stream,
            optimizer=FakeOptimizer(),
            loss_fn=FakeLoss(),
            num_epochs=10,
            hooks=[_StopHook(stop_after=3)],
        )
        loop.run(state)
        assert state.current_epoch == 3

    def test_does_not_exceed_num_epochs(self):
        """Boundary: even without early stop, epochs never exceed num_epochs."""
        from pipeline.training.train_loop import TrainLoop
        from tests.unit.conftest import FakeDataStream, FakeLoss, FakeModel, FakeOptimizer

        state = _make_state()
        stream = FakeDataStream(
            [
                Batch(inputs=np.array([[1.0]]), targets=np.array([[2.0]])),
            ]
        )
        loop = TrainLoop(
            model=FakeModel(),
            data_stream=stream,
            optimizer=FakeOptimizer(),
            loss_fn=FakeLoss(),
            num_epochs=3,
            hooks=[],
        )
        loop.run(state)
        assert state.current_epoch <= 3
        assert len(state.history["loss"]) == 3 * 1

    def test_hook_exception_does_not_kill_loop(self):
        """Error recovery: a crashing hook is logged but loop continues."""
        from pipeline.training.train_loop import TrainLoop
        from tests.unit.conftest import FakeDataStream, FakeLoss, FakeModel, FakeOptimizer

        state = _make_state()
        stream = FakeDataStream(
            [
                Batch(inputs=np.array([[1.0]]), targets=np.array([[2.0]])),
            ]
        )

        class CrashingHook(BaseHook):
            def on_batch_end(self, batch, loss, state):
                raise RuntimeError("hook crash")

        loop = TrainLoop(
            model=FakeModel(),
            data_stream=stream,
            optimizer=FakeOptimizer(),
            loss_fn=FakeLoss(),
            num_epochs=2,
            hooks=[CrashingHook()],
        )
        # Should not raise — hooks are non-critical
        loop.run(state)
        assert state.current_epoch == 2


# ---------------------------------------------------------------------------
# Gradient update test (with FakeOptimizer that actually mutates params)
# ---------------------------------------------------------------------------


class TestTrainLoopGradientUpdate:
    """TrainLoop runs the full forward-loss-backward-step cycle."""

    def test_optimizer_step_called(self):
        """Happy Path: optimizer.step() is called per batch."""
        from pipeline.training.train_loop import TrainLoop
        from tests.unit.conftest import FakeDataStream, FakeLoss, FakeModelWithParams, FakeOptimizer

        state = _make_state()
        stream = FakeDataStream(
            [
                Batch(inputs=np.array([[1.0, 2.0, 3.0, 4.0]]), targets=np.array([[0.5, 1.5]])),
            ]
        )
        opt = FakeOptimizer()
        loop = TrainLoop(
            model=FakeModelWithParams(in_features=4, out_features=2),
            data_stream=stream,
            optimizer=opt,
            loss_fn=FakeLoss(),
            num_epochs=3,
            hooks=[],
        )
        loop.run(state)
        assert opt.step_count == 3  # 3 epochs x 1 batch


# ---------------------------------------------------------------------------
# Frozen model (no parameters) tests
# ---------------------------------------------------------------------------


class _NoParamModel(BaseHook):
    """Shim — placeholder to remind: ModelProtocol must have forward() etc.

    We reuse FakeModel but test with an empty parameter list.
    """


class TestTrainLoopNoParams:
    """TrainLoop handles models without parameters (e.g., sklearn)."""

    def test_runs_with_empty_parameters(self):
        """Boundary: model returning empty parameters still runs."""
        from pipeline.training.train_loop import TrainLoop
        from tests.unit.conftest import (
            FakeDataStream,
            FakeLoss,
            ModelProtocol,
            OptimizerProtocol,
        )

        class NoParamModel(ModelProtocol):
            def forward(self, inputs):
                return inputs * 2

            def parameters(self):
                return iter([])  # empty

            def train_mode(self):
                pass

            def eval_mode(self):
                pass

        class _NoOpOptimizer(OptimizerProtocol):
            def step(self):
                pass

            def zero_grad(self):
                pass

        state = _make_state()
        stream = FakeDataStream(
            [
                Batch(inputs=np.array([[1.0]]), targets=np.array([[2.0]])),
            ]
        )
        loop = TrainLoop(
            model=NoParamModel(),
            data_stream=stream,
            optimizer=_NoOpOptimizer(),
            loss_fn=FakeLoss(),
            num_epochs=2,
            hooks=[],
        )
        loop.run(state)
        assert state.current_epoch == 2
        assert len(state.history["loss"]) == 2


# ---------------------------------------------------------------------------
# Edge cases: empty data stream
# ---------------------------------------------------------------------------


class TestTrainLoopEmptyStream:
    """TrainLoop handles data stream with zero batches."""

    def test_empty_stream_produces_no_loss_entries(self):
        """Boundary: empty data stream results in empty history per epoch."""
        from pipeline.training.train_loop import TrainLoop
        from tests.unit.conftest import FakeDataStream, FakeLoss, FakeModel, FakeOptimizer

        state = _make_state()
        stream = FakeDataStream([])
        loop = TrainLoop(
            model=FakeModel(),
            data_stream=stream,
            optimizer=FakeOptimizer(),
            loss_fn=FakeLoss(),
            num_epochs=3,
            hooks=[],
        )
        loop.run(state)
        assert len(state.history["loss"]) == 0


# ---------------------------------------------------------------------------
# Edge cases: loss = Loss object (not float)
# ---------------------------------------------------------------------------


class TestTrainLoopWithLossObject:
    """TrainLoop works with loss function returning Loss objects (with .backward())."""

    def test_accepts_loss_object(self):
        """Happy Path: LossProtocol that returns Loss object works."""
        from pipeline.training.train_loop import TrainLoop
        from tests.unit.conftest import (
            FakeDataStream,
            FakeModel,
            LossProtocol,
            OptimizerProtocol,
        )

        class _LossWithBackward(LossProtocol):
            def forward(self, predictions, targets):
                value = float(np.mean((np.asarray(predictions) - np.asarray(targets)) ** 2))
                return Loss(value=value, _backward_fn=lambda: None)

        class _CountOptimizer(OptimizerProtocol):
            def __init__(self):
                self.step_count = 0
                self.zero_count = 0

            def step(self):
                self.step_count += 1

            def zero_grad(self):
                self.zero_count += 1

        state = _make_state()
        stream = FakeDataStream(
            [
                Batch(inputs=np.array([[1.0]]), targets=np.array([[2.0]])),
            ]
        )
        opt = _CountOptimizer()
        loop = TrainLoop(
            model=FakeModel(),
            data_stream=stream,
            optimizer=opt,
            loss_fn=_LossWithBackward(),
            num_epochs=2,
            hooks=[],
        )
        loop.run(state)
        assert opt.step_count == 2
        assert opt.zero_count == 2
        assert len(state.history["loss"]) == 2
