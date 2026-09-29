"""Pytest hooks.

Missing PyMFEM is an error. Set ``B3_MICROMECH_ALLOW_NO_MFEM=1`` to skip
``mfem``-marked tests instead (local use without the solver).
"""

from __future__ import annotations

import os

import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "mfem: requires PyMFEM")
    config.addinivalue_line("markers", "b3tex: requires b3_tex")


def pytest_collection_modifyitems(config, items):
    try:
        import mfem  # noqa: F401
    except ImportError:
        if os.environ.get("B3_MICROMECH_ALLOW_NO_MFEM") == "1":
            skip = pytest.mark.skip(reason="mfem not installed")
            for item in items:
                if "mfem" in item.keywords:
                    item.add_marker(skip)
            return
        raise pytest.UsageError(
            "mfem is not installed. Set B3_MICROMECH_ALLOW_NO_MFEM=1 to skip mfem tests."
        )
