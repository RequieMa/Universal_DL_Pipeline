"""CSV export utility.

Writes numpy arrays and array-like objects to CSV files via pandas.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def to_csv(
    array: Any,
    path: str | Path,
    columns: list[str] | None = None,
) -> None:
    """Write a numpy array or array-like to a CSV file.

    Args:
        array: Array-like data to write. 1D arrays are reshaped to 2D.
        path: Output file path. Parent directories are created if needed.
        columns: Optional column names for the CSV header. Defaults to
            sequential integers.

    Raises:
        OSError: If the file cannot be written.
    """
    import pandas as pd

    arr = np.asarray(array)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(arr)
    if columns is not None:
        df.columns = columns
    df.to_csv(path, index=False)
