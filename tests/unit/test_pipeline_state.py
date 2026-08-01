"""Unit tests for pipeline.pipeline — PipelineState only."""

import numpy as np
import pytest

from pipeline.config import Config
from pipeline.evaluation.metrics import Metrics, accuracy, f1_score
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
        assert state.val_data_stream is None
        assert state.features is None
        assert state.model is None
        assert state.loss_fn is None
        assert state.optimizer is None
        assert state.history is None
        assert len(state.metrics) == 0

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
        state.data_stream = "fake_stream"
        assert state.data_stream == "fake_stream"

    def test_assign_model(self):
        """Happy Path: model field is writable and readable."""
        state = PipelineState(config=Config())
        state.model = "fake_model"
        assert state.model == "fake_model"

    def test_assign_metrics(self):
        """Happy Path: metrics field is a replaceable Metrics instance."""
        state = PipelineState(config=Config())
        state.metrics = Metrics(accuracy=accuracy, f1=f1_score)
        y_true = np.array([0, 1, 0])
        y_pred = np.array([0, 1, 0])
        state.metrics.compute(y_true, y_pred)
        assert state.metrics["accuracy"] == 1.0
        assert state.metrics["f1"] == 1.0

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
        y_true = np.array([0, 1, 0])
        y_pred1 = np.array([0, 1, 0])
        y_pred2 = np.array([1, 1, 0])

        state1 = PipelineState(config=cfg, mode="train")
        state2 = PipelineState(config=cfg, mode="infer")

        state1.metrics = Metrics(accuracy=accuracy)
        state2.metrics = Metrics(accuracy=accuracy)
        state1.metrics.compute(y_true, y_pred1)
        state2.metrics.compute(y_true, y_pred2)

        assert state1.metrics["accuracy"] == 1.0
        assert state2.metrics["accuracy"] == 2.0 / 3.0
        assert state1.mode != state2.mode

    def test_metrics_default_dicts_isolated(self):
        """Concurrency: default Metrics instance is unique per PipelineState."""
        s1 = PipelineState(config=Config())
        s2 = PipelineState(config=Config())
        assert s1.metrics is not s2.metrics


class TestPipelineStatePhase1Amendments:
    """Tests for Phase 1 additions to PipelineState."""

    def test_val_data_stream_defaults_to_none(self):
        """Happy Path: val_data_stream defaults to None."""
        state = PipelineState(config=Config())
        assert state.val_data_stream is None

    def test_val_data_stream_is_writable(self):
        """Happy Path: val_data_stream accepts a DataStream."""
        state = PipelineState(config=Config())
        state.val_data_stream = "fake_val_stream"
        assert state.val_data_stream == "fake_val_stream"

    def test_metrics_defaults_to_empty_metrics_instance(self):
        """Happy Path: metrics defaults to an empty Metrics instance."""
        state = PipelineState(config=Config())
        assert len(state.metrics) == 0
        assert isinstance(state.metrics, Metrics)

    def test_metrics_is_replaceable(self):
        """Happy Path: metrics can be replaced with a configured Metrics."""
        state = PipelineState(config=Config())
        state.metrics = Metrics(accuracy=accuracy, f1=f1_score)
        assert len(state.metrics) == 2
        assert "accuracy" in state.metrics
        assert "f1" in state.metrics
