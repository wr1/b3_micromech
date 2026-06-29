import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "mfem: requires PyMFEM")


def pytest_collection_modifyitems(config, items):
    try:
        import mfem  # noqa: F401
    except ImportError:
        skip = pytest.mark.skip(reason="mfem not installed")
        for item in items:
            if "mfem" in item.keywords:
                item.add_marker(skip)
