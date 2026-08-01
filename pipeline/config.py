"""Configuration dataclass for the pipeline.

A single source of truth for all pipeline settings. Serialized
to/from YAML files. Every field has a sensible default so students
can start with ``Config()`` and add options as they learn them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


def _validate_positive(value: int | float, name: str) -> None:
    """Raise ValueError if value is not positive."""
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")


def _validate_non_negative(value: int | float, name: str) -> None:
    """Raise ValueError if value is negative."""
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value}")


def _validate_ratio(value: float, name: str) -> None:
    """Raise ValueError if not in (0, 1)."""
    if not (0.0 < value < 1.0):
        raise ValueError(f"{name} must be in (0, 1), got {value}")


@dataclass
class Config:
    """Pipeline configuration.

    All fields have defaults so beginners can start minimal and add
    options as they progress through the syllabus.

    Attributes:
        data_dir: Root directory for datasets.
        output_dir: Where checkpoints and predictions are written.
        task_name: Human-readable label for this experiment.
        seed: Random seed for reproducibility.
        batch_size: Samples per training batch.
        num_epochs: Maximum training epochs.
        learning_rate: Initial learning rate for the optimizer.
        train_ratio: Fraction of data used for training (remainder = validation).
        device: Compute device -- ``"auto"``, ``"cpu"``, ``"cuda"``, or ``"mps"``.
        use_amp: Enable automatic mixed precision (GPU only).
        num_workers: DataLoader worker processes.
        model_name: Registry key for the model backbone.
        optimizer_name: Registry key for the optimizer.
        loss_name: Registry key for the loss function.
        metrics: List of metric names to compute during evaluation.
        checkpoint_dir: Directory for model checkpoints (None = output_dir/checkpoints).
        log_interval: Log training progress every N batches.
    """

    # -- Paths ----------------------------------------------------------------
    data_dir: str = "./data"
    output_dir: str = "./outputs"
    task_name: str = ""

    # -- Reproducibility ------------------------------------------------------
    seed: int = 42

    # -- Data -----------------------------------------------------------------
    batch_size: int = 32
    train_ratio: float = 0.8

    # -- Training -------------------------------------------------------------
    num_epochs: int = 10
    learning_rate: float = 0.001

    # -- Hardware -------------------------------------------------------------
    device: str = "auto"
    use_amp: bool = False
    num_workers: int = 0

    # -- Components (registry keys) -------------------------------------------
    model_name: str = ""
    optimizer_name: str = "sgd"
    loss_name: str = "cross_entropy"
    metrics: list[str] = field(default_factory=lambda: ["accuracy"])
    # WHY: field(default_factory=...) creates a fresh list per instance,
    # preventing the mutable-default-argument footgun.

    # -- Checkpointing --------------------------------------------------------
    checkpoint_dir: str | None = None

    # -- Logging --------------------------------------------------------------
    log_interval: int = 10

    def __post_init__(self) -> None:
        """Validate field values after dataclass construction."""
        _validate_positive(self.batch_size, "batch_size")
        _validate_non_negative(self.num_epochs, "num_epochs")
        _validate_positive(self.log_interval, "log_interval")
        _validate_positive(self.learning_rate, "learning_rate")
        _validate_ratio(self.train_ratio, "train_ratio")

    @classmethod
    def from_yaml(cls, path: str) -> Config:
        """Load configuration from a YAML file.

        Args:
            path: Path to a ``.yaml`` or ``.yml`` file.

        Returns:
            A new Config with values from the file (unspecified
            fields retain their defaults).

        Raises:
            FileNotFoundError: If the file does not exist.
        """
        yaml_path = Path(path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        # WHY: YAML import is inside the method rather than at module
        # level, so Config is usable without PyYAML installed. Only
        # users who need YAML loading need the dependency.
        import yaml  # type: ignore[import-untyped]

        with open(yaml_path) as f:
            data = yaml.safe_load(f)

        if data is None:
            data = {}

        # Filter to only known fields (ignore extras silently -- forward compat)
        known = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in known}
        return cls(**filtered)
