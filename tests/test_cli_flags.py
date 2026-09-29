"""Boolean CLI options take no value."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "b3_micromech.cli", *args],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def test_no_disk_cache_is_a_flag():
    result = _run("register-fea-micromech", "--no-disk-cache", "--name", "flag_probe")
    combined = result.stdout + result.stderr
    assert "expected one argument" not in combined
    assert result.returncode == 0
    assert "disk cache" in combined
    assert "enabled=False" in combined


@pytest.mark.mfem
def test_solve_plot_is_a_flag(tmp_path: Path):
    out = tmp_path / "out"
    result = _run(
        "solve",
        str(EXAMPLES / "ud_transverse.yaml"),
        "--plot",
        "--out",
        str(out),
    )
    combined = result.stdout + result.stderr
    assert "expected one argument" not in combined
    assert result.returncode == 0, combined
    assert (out / "C_eff.npz").is_file()
    assert any((out / "plots").glob("*.png"))
