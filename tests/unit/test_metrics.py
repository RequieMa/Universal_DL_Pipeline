"""Unit tests for pipeline.evaluation.metrics — Metrics class and pure functions."""
import numpy as np
import pytest

from pipeline.evaluation.metrics import (
    Metrics,
    accuracy,
    confusion_matrix,
    f1_score,
    precision,
    recall,
)


class TestAccuracy:
    """Tests for accuracy()."""

    def test_perfect_accuracy(self):
        """Happy Path: identical predictions → 1.0."""
        y_true = np.array([0, 1, 2, 1, 0])
        y_pred = np.array([0, 1, 2, 1, 0])
        assert accuracy(y_true, y_pred) == 1.0

    def test_zero_accuracy(self):
        """Happy Path: all wrong → 0.0."""
        y_true = np.array([0, 0, 0])
        y_pred = np.array([1, 1, 1])
        assert accuracy(y_true, y_pred) == 0.0

    def test_partial_accuracy(self):
        """Happy Path: 3/5 correct → 0.6."""
        y_true = np.array([0, 1, 2, 1, 0])
        y_pred = np.array([0, 1, 0, 1, 1])
        assert accuracy(y_true, y_pred) == 0.6

    def test_binary_accuracy(self):
        """Happy Path: binary classification."""
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 1])
        assert accuracy(y_true, y_pred) == 0.75


class TestPrecision:
    """Tests for precision()."""

    def test_perfect_precision(self):
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1])
        assert precision(y_true, y_pred) == 1.0

    def test_precision_with_false_positives(self):
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([1, 1, 0, 1])
        # TP=2, FP=1, precision=2/3
        assert precision(y_true, y_pred) == pytest.approx(2 / 3)

    def test_precision_macro_multiclass(self):
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 2, 2, 0, 1, 1])
        result = precision(y_true, y_pred, average="macro")
        # class 0: TP=1, FP=0, prec=1.0
        # class 1: TP=1, FP=1, prec=0.5
        # class 2: TP=1, FP=1, prec=0.5
        # macro = (1.0 + 0.5 + 0.5) / 3 approx 0.667
        assert 0.66 < result < 0.67

    def test_precision_zero_division(self):
        """Boundary: when no positive predictions, precision is 0.0."""
        y_true = np.array([0, 1, 0])
        y_pred = np.array([0, 0, 0])
        result = precision(y_true, y_pred)
        assert result == 0.0


class TestRecall:
    """Tests for recall()."""

    def test_perfect_recall(self):
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1])
        assert recall(y_true, y_pred) == 1.0

    def test_recall_with_false_negatives(self):
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 0, 0, 1])
        # TP=1, FN=1, recall=1/2
        assert recall(y_true, y_pred) == 0.5

    def test_recall_macro_multiclass(self):
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 2, 2, 0, 1, 1])
        result = recall(y_true, y_pred, average="macro")
        # class 0: TP=2, FN=0, rec=1.0
        # class 1: TP=1, FN=1, rec=0.5
        # class 2: TP=1, FN=1, rec=0.5
        # macro = (1.0 + 0.5 + 0.5) / 3 approx 0.667
        assert 0.66 < result < 0.67

    def test_recall_zero_division(self):
        """Boundary: when no true positives, recall is 0.0."""
        y_true = np.array([0, 0, 0])
        y_pred = np.array([1, 1, 1])
        result = recall(y_true, y_pred)
        assert result == 0.0


class TestF1Score:
    """Tests for f1_score()."""

    def test_perfect_f1(self):
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1])
        assert f1_score(y_true, y_pred) == 1.0

    def test_f1_harmonic_mean(self):
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([1, 1, 0, 1])
        # prec=2/3, rec=2/2=1.0, f1=2*(2/3*1)/(2/3+1)=0.8
        assert f1_score(y_true, y_pred) == pytest.approx(0.8)

    def test_f1_zero_when_precision_zero(self):
        """Boundary: zero precision → zero F1."""
        y_true = np.array([1, 1, 1])
        y_pred = np.array([0, 0, 0])
        assert f1_score(y_true, y_pred) == 0.0

    def test_f1_macro_multiclass(self):
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 2, 2, 0, 1, 1])
        result = f1_score(y_true, y_pred, average="macro")
        # class 0: p=1.0, r=1.0, f1=1.0
        # class 1: p=0.5, r=0.5, f1=0.5
        # class 2: p=0.5, r=0.5, f1=0.5
        # macro = (1.0 + 0.5 + 0.5) / 3 approx 0.667
        assert 0.66 < result < 0.67


class TestConfusionMatrix:
    """Tests for confusion_matrix()."""

    def test_perfect_confusion_matrix(self):
        y_true = np.array([0, 1, 2])
        y_pred = np.array([0, 1, 2])
        cm = confusion_matrix(y_true, y_pred)
        expected = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        assert np.array_equal(cm, expected)

    def test_confusion_matrix_with_errors(self):
        y_true = np.array([0, 0, 1, 1, 2, 2])
        y_pred = np.array([0, 1, 1, 1, 2, 0])
        cm = confusion_matrix(y_true, y_pred)
        # y_true=0, y_pred=0 → (0,0)+=1
        # y_true=0, y_pred=1 → (0,1)+=1
        # y_true=1, y_pred=1 → (1,1)+=2
        # y_true=2, y_pred=2 → (2,2)+=1
        # y_true=2, y_pred=0 → (2,0)+=1
        expected = np.array([[1, 1, 0], [0, 2, 0], [1, 0, 1]])
        assert np.array_equal(cm, expected)

    def test_confusion_matrix_binary(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 1, 1, 1])
        cm = confusion_matrix(y_true, y_pred)
        expected = np.array([[1, 1], [0, 2]])
        assert np.array_equal(cm, expected)

    def test_confusion_matrix_auto_num_classes(self):
        y_true = np.array([0, 2, 4])
        y_pred = np.array([0, 2, 4])
        cm = confusion_matrix(y_true, y_pred)
        assert cm.shape == (5, 5)  # 0..4 inclusive = 5 classes


class TestMetrics:
    """Tests for the Metrics container class."""

    def test_metrics_empty(self):
        """Boundary: empty Metrics is valid."""
        m = Metrics()
        assert len(m) == 0
        assert list(m) == []

    def test_metrics_registered_names(self):
        """Happy Path: __iter__ yields registered metric names."""
        m = Metrics(accuracy=accuracy, f1=f1_score)
        assert set(m) == {"accuracy", "f1"}
        assert len(m) == 2

    def test_metrics_contains(self):
        """Happy Path: __contains__ checks registered names."""
        m = Metrics(accuracy=accuracy)
        assert "accuracy" in m
        assert "f1" not in m

    def test_compute_returns_dict(self):
        """Happy Path: compute returns {name: value} dict."""
        m = Metrics(acc=accuracy)
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1])
        result = m.compute(y_true, y_pred)
        assert result == {"acc": 1.0}

    def test_compute_multiple_metrics(self):
        """Happy Path: compute runs all registered metrics."""
        m = Metrics(acc=accuracy, f1=f1_score)
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 1])
        result = m.compute(y_true, y_pred)
        assert "acc" in result
        assert "f1" in result
        assert 0.0 <= result["acc"] <= 1.0
        assert 0.0 <= result["f1"] <= 1.0

    def test_getitem_after_compute(self):
        """Happy Path: __getitem__ returns computed value."""
        m = Metrics(acc=accuracy)
        y_true = np.array([0, 1])
        y_pred = np.array([0, 1])
        m.compute(y_true, y_pred)
        assert m["acc"] == 1.0

    def test_getitem_before_compute_raises(self):
        """Error: __getitem__ raises KeyError before compute()."""
        m = Metrics(acc=accuracy)
        with pytest.raises(KeyError):
            m["acc"]

    def test_getitem_unregistered_name_raises(self):
        """Error: __getitem__ raises KeyError for unknown metric."""
        m = Metrics(acc=accuracy)
        y_true = np.array([0, 1])
        y_pred = np.array([0, 1])
        m.compute(y_true, y_pred)
        with pytest.raises(KeyError):
            m["nonexistent"]
