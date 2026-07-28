"""Unit tests for pipeline.pipeline — PipelineState and BasePipeline."""

import numpy as np
import pytest

from pipeline.config import Config
from pipeline.pipeline import PipelineState


class TestPipelineStateDefaults:
    """Tests for PipelineState default values."""

    def test_create_with_config_only(self):
        """Happy Path: PipelineState(config) uses all defaults."""
        cfg = Config()
        state = PipelineState(config=cfg)
        assert state.config is cfg
        assert state.mode == "train"
        assert state.data_stream is None
        assert state.features is None
        assert state.model is None
        assert state.loss_fn is None
        assert state.optimizer is None
        assert state.history is None
        assert state.metrics == {}
        assert state.predictions is None
        assert state.current_epoch == 0
        assert state.should_stop is False

    def test_create_with_infer_mode(self):
        """Happy Path: PipelineState with mode='infer'."""
        state = PipelineState(config=Config(), mode="infer")
        assert state.mode == "infer"


class TestPipelineStateProperties:
    """Tests for PipelineState @property methods."""

    def test_is_training_when_train_mode(self):
        """Property: is_training is True when mode='train'."""
        state = PipelineState(config=Config(), mode="train")
        assert state.is_training is True

    def test_is_training_false_when_infer_mode(self):
        """Property: is_training is False when mode='infer'."""
        state = PipelineState(config=Config(), mode="infer")
        assert state.is_training is False

    def test_is_training_read_only(self):
        """Property: is_training is read-only (raises AttributeError on set)."""
        state = PipelineState(config=Config())
        with pytest.raises(AttributeError):
            state.is_training = False  # type: ignore[misc]


class TestPipelineStateFieldAssignment:
    """Tests for PipelineState field mutations."""

    def test_assign_and_read_data_stream(self):
        """Happy Path: data_stream field is writable and readable."""
        state = PipelineState(config=Config())
        state.data_stream = "fake_stream"  # protocol duck-type
        assert state.data_stream == "fake_stream"

    def test_assign_model(self):
        """Happy Path: model field is writable and readable."""
        state = PipelineState(config=Config())
        state.model = "fake_model"  # protocol duck-type
        assert state.model == "fake_model"

    def test_assign_metrics(self):
        """Happy Path: metrics dict is mutable."""
        state = PipelineState(config=Config())
        state.metrics["accuracy"] = 0.95
        state.metrics["f1"] = 0.93
        assert state.metrics == {"accuracy": 0.95, "f1": 0.93}

    def test_should_stop_flag(self):
        """Happy Path: should_stop is settable (for early stopping hooks)."""
        state = PipelineState(config=Config())
        assert state.should_stop is False
        state.should_stop = True
        assert state.should_stop is True

    def test_current_epoch_increment(self):
        """Happy Path: current_epoch is mutable for training loop."""
        state = PipelineState(config=Config())
        for epoch in range(5):
            state.current_epoch = epoch
        assert state.current_epoch == 4

    def test_predictions_assign_arraylike(self):
        """Happy Path: predictions field accepts numpy array."""
        state = PipelineState(config=Config())
        preds = np.array([[0.1, 0.9], [0.8, 0.2]])
        state.predictions = preds
        assert np.array_equal(state.predictions, preds)


class TestPipelineStateIsolation:
    """Tests for PipelineState isolation between runs."""

    def test_two_states_independent(self):
        """Concurrency: two PipelineState instances don't share state."""
        cfg = Config()
        state1 = PipelineState(config=cfg, mode="train")
        state2 = PipelineState(config=cfg, mode="infer")

        state1.metrics["accuracy"] = 0.9
        state2.metrics["accuracy"] = 0.5

        assert state1.metrics["accuracy"] == 0.9
        assert state2.metrics["accuracy"] == 0.5
        assert state1.mode != state2.mode

    def test_metrics_default_dicts_isolated(self):
        """Concurrency: default metrics dict is unique per instance."""
        s1 = PipelineState(config=Config())
        s2 = PipelineState(config=Config())
        s1.metrics["x"] = 1.0
        assert "x" not in s2.metrics
