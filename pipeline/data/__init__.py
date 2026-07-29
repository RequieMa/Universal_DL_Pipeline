"""Data sources and split utilities for the pipeline."""

from pipeline.data.csv_source import CsvDataSource

__all__ = ["CsvDataSource"]

# NOTE: train_test_split will be added in Task 7 (pipeline/data/split.py).
# Once that module exists, import it here as:
#   from pipeline.data.split import train_test_split
# and add "train_test_split" to __all__.
