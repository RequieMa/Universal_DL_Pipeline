"""Unit tests for pipeline.hooks — BaseHook."""
from pipeline.config import Config
from pipeline.hooks import BaseHook
from pipeline.pipeline import PipelineState


class TestBaseHook:
    """Tests for BaseHook."""

    def test_all_hooks_default_to_noop(self):
        """Happy Path: all BaseHook methods are no-ops by default.
        A bare BaseHook() instance should not crash on any event."""
        hook = BaseHook()
        state = PipelineState(config=Config())
        hook.on_stage_start("load_data", state)
        hook.on_stage_end("load_data", state)
        hook.on_epoch_start(0, state)
        hook.on_epoch_end(0, state)
        hook.on_batch_end(0, 0.5, state)

    def test_subclass_overrides_one_method(self):
        """Happy Path: subclass can override a single hook method."""
        called = False

        class OneMethodHook(BaseHook):
            def on_stage_start(self, stage, state):
                nonlocal called
                called = True

        hook = OneMethodHook()
        hook.on_stage_start("train", PipelineState(config=Config()))
        assert called is True

    def test_subclass_overrides_all_methods(self):
        """Happy Path: subclass can override all five hook methods."""
        log: list[str] = []

        class FullHook(BaseHook):
            def on_stage_start(self, stage, state):
                log.append(f"stage_start:{stage}")
            def on_stage_end(self, stage, state):
                log.append(f"stage_end:{stage}")
            def on_epoch_start(self, epoch, state):
                log.append(f"epoch_start:{epoch}")
            def on_epoch_end(self, epoch, state):
                log.append(f"epoch_end:{epoch}")
            def on_batch_end(self, batch, loss, state):
                log.append(f"batch:{batch}:{loss}")

        hook = FullHook()
        state = PipelineState(config=Config())
        hook.on_stage_start("train", state)
        hook.on_epoch_start(0, state)
        hook.on_batch_end(5, 0.3, state)
        hook.on_epoch_end(0, state)
        hook.on_stage_end("train", state)

        assert log == [
            "stage_start:train",
            "epoch_start:0",
            "batch:5:0.3",
            "epoch_end:0",
            "stage_end:train",
        ]

    def test_hook_reads_state_current_epoch(self):
        """Happy Path: hook can read state.current_epoch."""
        last_epoch: int = -1

        class EpochReaderHook(BaseHook):
            def on_epoch_end(self, epoch, state):
                nonlocal last_epoch
                last_epoch = state.current_epoch

        state = PipelineState(config=Config())
        state.current_epoch = 7
        EpochReaderHook().on_epoch_end(7, state)
        assert last_epoch == 7

    def test_hook_sets_should_stop(self):
        """Happy Path: hook can set state.should_stop for early stopping."""
        state = PipelineState(config=Config())
        assert state.should_stop is False

        class EarlyStopHook(BaseHook):
            def on_epoch_end(self, epoch, state):
                if epoch >= 5:
                    state.should_stop = True

        EarlyStopHook().on_epoch_end(5, state)
        assert state.should_stop is True

    def test_multiple_hooks_independent(self):
        """Concurrency: two hook instances don't share state."""
        counter_a = 0
        counter_b = 0

        class CounterHook(BaseHook):
            def __init__(self, tag: str):
                self.tag = tag
            def on_stage_start(self, stage, state):
                nonlocal counter_a, counter_b
                if self.tag == "a":
                    counter_a += 1  # type: ignore[unused-ignore]
                else:
                    counter_b += 1  # type: ignore[unused-ignore]

        a = CounterHook("a")
        b = CounterHook("b")
        state = PipelineState(config=Config())
        a.on_stage_start("load_data", state)
        b.on_stage_start("load_data", state)
        assert counter_a == 1
        assert counter_b == 1
