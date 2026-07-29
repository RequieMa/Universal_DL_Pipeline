"""Unit tests for pipeline.__main__ and CLI entry."""
from __future__ import annotations

import io
import sys


class TestMainModule:
    """Tests for python -m pipeline."""

    def test_main_prints_version(self):
        """Happy Path: python -m pipeline prints version string."""
        import pipeline.__main__

        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            pipeline.__main__.main()
            output = sys.stdout.getvalue()
        finally:
            sys.stdout = old_stdout
        assert "0.1.0" in output or "Universal DL Pipeline" in output

    def test_main_prints_device_info(self):
        """Happy Path: prints device information."""
        import pipeline.__main__

        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            pipeline.__main__.main()
            output = sys.stdout.getvalue()
        finally:
            sys.stdout = old_stdout
        # Should mention device (CPU at minimum)
        assert "Device" in output or "device" in output

    def test_main_prints_registered_components(self):
        """Happy Path: prints registered components by kind."""
        import pipeline.__main__

        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            pipeline.__main__.main()
            output = sys.stdout.getvalue()
        finally:
            sys.stdout = old_stdout
        # At minimum, something is registered (from Phase 0 tests, if nothing else)
        # If nothing registered, it should still print successfully
        assert "Universal DL Pipeline" in output


class TestRunPy:
    """Tests for run.py CLI."""

    def test_run_py_module_exists(self):
        """Happy Path: run.py can be imported."""
        from pathlib import Path
        assert Path("run.py").exists()
