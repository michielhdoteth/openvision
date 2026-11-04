"""Guards for the installed distribution.

`pyproject.toml` sets `pythonpath = ["."]` for pytest, which puts the project
root on `sys.path` for every test run. That hides a broken editable install: the
finder can be missing a package and the console script can be unusable while the
whole suite still passes from the repo root. The checks below run in a
subprocess whose cwd is outside the project tree, so cwd cannot mask a
regression.
"""
import os
import subprocess
import sys
import sysconfig
import tempfile
import textwrap
import tomllib
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _run_outside_project_root(code: str) -> subprocess.CompletedProcess:
    """Run a snippet in a fresh interpreter with no project root on sys.path."""
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        cwd=tempfile.gettempdir(),
        capture_output=True,
        text=True,
        timeout=120,
    )


def test_pyproject_package_discovery_includes_the_public_openvision_package():
    """`openvision*` must stay in the setuptools include list.

    It was dropped in 6c1c30e and re-added in 8889660. Dropping it again ships a
    distribution whose console script dies on `from openvision.models import ...`,
    and pytest will not notice.
    """
    data = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    include = data["tool"]["setuptools"]["packages"]["find"]["include"]
    assert "openvision*" in include, (
        "openvision* is missing from [tool.setuptools.packages.find].include; "
        f"found {include}"
    )


def test_installed_distribution_maps_the_public_openvision_package():
    """The editable finder must resolve `openvision` without help from cwd."""
    result = _run_outside_project_root("""
        from pathlib import Path

        import openvision

        assert Path(openvision.__file__).is_file(), openvision.__file__
        print("openvision-ok")
    """)
    assert result.returncode == 0, result.stderr
    assert "openvision-ok" in result.stdout


def test_console_script_entry_point_resolves_outside_the_project_root():
    """`cli.app:app` is what `openvision` runs, so it must import cleanly."""
    result = _run_outside_project_root("""
        from importlib.metadata import version

        from cli.app import app

        assert version("openvision")
        print("entry-point-ok")
    """)
    assert result.returncode == 0, result.stderr
    assert "entry-point-ok" in result.stdout


def test_cli_app_forces_utf8_console_for_the_console_script():
    """`openvision status` must not render mangled ASCII fallback boxes.

    The reconfigure used to live only in the root `__main__.py` shim, which the
    `[project.scripts] openvision = "cli.app:app"` entry point bypasses, so the
    console script rendered while `python __main__.py status` rendered properly.

    The test first forces the streams back to cp1252 so it cannot pass just
    because the ambient environment already defaults to UTF-8.
    """
    result = _run_outside_project_root("""
        import sys

        sys.stdout.reconfigure(encoding="cp1252", errors="replace")
        sys.stderr.reconfigure(encoding="cp1252", errors="replace")

        import cli.app  # noqa: F401

        print(sys.stdout.encoding.lower().replace("-", ""))
    """)
    assert result.returncode == 0, result.stderr
    assert "utf8" in result.stdout, result.stdout


def test_console_script_status_renders_unicode_box_characters():
    """The shipped console script must emit real box drawing, not ASCII `+---+`."""
    script = Path(sysconfig.get_path("scripts")) / ("openvision.exe" if os.name == "nt" else "openvision")
    if not script.exists():
        pytest.skip("openvision console script is not installed in this environment")

    result = subprocess.run(
        [str(script), "status"],
        cwd=tempfile.gettempdir(),
        capture_output=True,
        timeout=180,
    )
    out = result.stdout.decode("utf-8", "replace")
    assert result.returncode == 0, result.stderr.decode("utf-8", "replace")
    assert "\u250c" in out or "\u251c" in out or "\u2502" in out, out[:500]
    assert "\ufffd" not in out, "mojibake in console script output"
