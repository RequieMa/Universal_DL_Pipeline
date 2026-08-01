"""Unit tests for pipeline.config — Config dataclass."""

import pytest

from pipeline.config import Config


class TestConfigDefaults:
    """Tests for Config default values."""

    def test_create_with_no_args_uses_defaults(self):
        """Happy Path: Config() uses all documented defaults."""
        cfg = Config()
        assert cfg.data_dir == "./data"
        assert cfg.output_dir == "./outputs"
        assert cfg.seed == 42
        assert cfg.batch_size == 32
        assert cfg.num_epochs == 10
        assert cfg.learning_rate == pytest.approx(0.001)
        assert cfg.train_ratio == pytest.approx(0.8)
        assert cfg.device == "auto"
        assert cfg.use_amp is False
        assert cfg.num_workers == 0
        assert cfg.model_name == ""
        assert cfg.optimizer_name == "sgd"
        assert cfg.loss_name == "cross_entropy"
        assert cfg.metrics == ["accuracy"]
        assert cfg.log_interval == 10

    def test_task_name_defaults_to_empty(self):
        """Boundary: task_name default is empty string."""
        cfg = Config()
        assert cfg.task_name == ""

    def test_checkpoint_dir_defaults_to_none(self):
        """Boundary: checkpoint_dir default is None."""
        cfg = Config()
        assert cfg.checkpoint_dir is None


class TestConfigOverride:
    """Tests for Config field overrides."""

    def test_override_single_field(self):
        """Happy Path: override one field keeps defaults for others."""
        cfg = Config(batch_size=64)
        assert cfg.batch_size == 64
        assert cfg.num_epochs == 10  # default preserved

    def test_override_all_fields(self):
        """Happy Path: override every field."""
        cfg = Config(
            data_dir="/custom/data",
            output_dir="/custom/output",
            task_name="test_task",
            seed=123,
            batch_size=128,
            num_epochs=50,
            learning_rate=0.01,
            train_ratio=0.9,
            device="cuda",
            use_amp=True,
            num_workers=4,
            model_name="resnet18",
            optimizer_name="adam",
            loss_name="mse",
            metrics=["accuracy", "f1"],
            checkpoint_dir="/custom/ckpt",
            log_interval=5,
        )
        assert cfg.data_dir == "/custom/data"
        assert cfg.seed == 123
        assert cfg.batch_size == 128
        assert cfg.device == "cuda"
        assert cfg.use_amp is True
        assert cfg.model_name == "resnet18"
        assert cfg.optimizer_name == "adam"
        assert cfg.loss_name == "mse"
        assert cfg.metrics == ["accuracy", "f1"]
        assert cfg.checkpoint_dir == "/custom/ckpt"
        assert cfg.log_interval == 5


class TestConfigValidation:
    """Tests for Config validation on invalid inputs."""

    def test_negative_batch_size_raises(self):
        """Boundary: negative batch_size raises ValueError."""
        with pytest.raises(ValueError, match="batch_size"):
            Config(batch_size=-1)

    def test_zero_batch_size_raises(self):
        """Boundary: batch_size=0 raises ValueError."""
        with pytest.raises(ValueError, match="batch_size"):
            Config(batch_size=0)

    def test_negative_num_epochs_raises(self):
        """Boundary: negative num_epochs raises ValueError."""
        with pytest.raises(ValueError, match="num_epochs"):
            Config(num_epochs=-1)

    def test_train_ratio_zero_raises(self):
        """Boundary: train_ratio=0 raises ValueError."""
        with pytest.raises(ValueError, match="train_ratio"):
            Config(train_ratio=0.0)

    def test_train_ratio_one_raises(self):
        """Boundary: train_ratio=1.0 raises ValueError."""
        with pytest.raises(ValueError, match="train_ratio"):
            Config(train_ratio=1.0)

    def test_train_ratio_out_of_range_raises(self):
        """Boundary: train_ratio > 1.0 raises ValueError."""
        with pytest.raises(ValueError, match="train_ratio"):
            Config(train_ratio=1.5)

    def test_negative_learning_rate_raises(self):
        """Boundary: negative learning_rate raises ValueError."""
        with pytest.raises(ValueError, match="learning_rate"):
            Config(learning_rate=-0.001)

    def test_negative_log_interval_raises(self):
        """Boundary: negative log_interval raises ValueError."""
        with pytest.raises(ValueError, match="log_interval"):
            Config(log_interval=-1)


class TestConfigImmutability:
    """Tests for Config field-type safety."""

    def test_fields_are_accessible(self):
        """Happy Path: all fields readable after construction."""
        cfg = Config(task_name="mnist", batch_size=64)
        assert cfg.task_name == "mnist"

    def test_metrics_mutation_shared_list_regression(self):
        """Concurrency: two Configs with default metrics don't share the
        same list object (common dataclass footgun)."""
        cfg1 = Config()
        cfg2 = Config()
        cfg1.metrics.append("f1")
        assert "f1" not in cfg2.metrics
        assert cfg2.metrics == ["accuracy"]


class TestConfigFromYaml:
    """Tests for Config.from_yaml() classmethod."""

    def test_from_yaml_loads_config(self, tmp_path):
        """Happy Path: from_yaml loads a minimal config file."""
        yaml_path = tmp_path / "config.yaml"
        yaml_path.write_text("""
task_name: my_task
batch_size: 64
num_epochs: 20
learning_rate: 0.01
""")
        cfg = Config.from_yaml(str(yaml_path))
        assert cfg.task_name == "my_task"
        assert cfg.batch_size == 64
        assert cfg.num_epochs == 20
        assert cfg.learning_rate == pytest.approx(0.01)
        # Unspecified fields use defaults
        assert cfg.seed == 42

    def test_from_yaml_file_not_found(self, tmp_path):
        """Error recovery: from_yaml on missing file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            Config.from_yaml(str(tmp_path / "nonexistent.yaml"))
