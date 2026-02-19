#!/usr/bin/env python3
"""
Test runner script for the JSSP GNN project.

The script wraps the pytest invocation to ensure that we consistently use the
`uv` package manager (when available) and expose a friendly CLI for filtering
tests, toggling coverage, enabling parallel execution, and emitting CI friendly
reports.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from collections.abc import Mapping, MutableMapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path


DEFAULT_MARKERS = {
    "unit": "unit",
    "integration": "integration",
    "fast": "not slow",
    "slow": "slow",
    "gpu": "gpu",
}
DEFAULT_UV_CACHE = Path(".uv_cache")


@dataclass(frozen=True)
class TestOptions:
    """Container describing how pytest should be executed."""

    test_type: str = "all"
    coverage: bool = False
    parallel: bool = False
    verbose: bool = False
    failed: bool = False
    junit_xml: Path | None = None
    pytest_args: Sequence[str] = field(default_factory=tuple)

    def marker_expression(self) -> str | None:
        """
        Return the marker expression that should be applied for the selected
        test type. The value is `None` when every test should be executed.
        """

        return DEFAULT_MARKERS.get(self.test_type)


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments and expose the result as a namespace."""

    parser = argparse.ArgumentParser(description="Run tests for JSSP GNN project")
    parser.add_argument(
        "--type",
        choices=["unit", "integration", "all", "fast", "slow", "gpu"],
        default="all",
        help="Type of tests to run",
    )
    parser.add_argument(
        "--coverage",
        action="store_true",
        help="Enable coverage reporting when running pytest",
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Run tests in parallel (requires pytest-xdist)",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument(
        "--failed",
        action="store_true",
        help="Run only previously failed tests (pytest --lf)",
    )
    parser.add_argument(
        "--junit",
        type=Path,
        help="Optional path to write JUnit XML results (useful for CI summaries)",
    )
    parser.add_argument(
        "--runner",
        choices=["auto", "uv", "python"],
        default="auto",
        help="Select whether pytest should run through uv or the system interpreter",
    )
    parser.add_argument(
        "pytest_args",
        nargs=argparse.REMAINDER,
        help="Extra arguments passed directly to pytest (prefix with --)",
    )
    return parser.parse_args(argv)


def normalize_pytest_args(pytest_args: Sequence[str]) -> tuple[str, ...]:
    """
    Remove the stand-alone ``--`` marker that argparse retains when forwarding
    additional positional arguments to pytest.
    """

    if pytest_args and pytest_args[0] == "--":
        return tuple(pytest_args[1:])
    return tuple(pytest_args)


def ensure_uv_runner(
    cache_dir: Path,
) -> tuple[list[str], MutableMapping[str, str]] | None:
    """
    Return the uv pytest command (and environment) if uv is available locally.
    The cache directory is created inside the repository to avoid permission
    issues on environments where the default ~/.cache path is locked down.
    """

    uv_path = shutil.which("uv")
    if uv_path is None:
        return None

    env = os.environ.copy()
    cache_location = Path(env.get("UV_CACHE_DIR", cache_dir))
    try:
        cache_location.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise RuntimeError(
            f"Unable to create uv cache dir {cache_location}: {error}"
        ) from error

    env.setdefault("UV_CACHE_DIR", str(cache_location))
    return [uv_path, "run", "-m", "pytest"], env


def resolve_base_command(preferred: str) -> tuple[list[str], Mapping[str, str]]:
    """
    Determine which runner should execute pytest. When `uv` is preferred we try
    to build an `uv run -m pytest` command and gracefully fall back to the
    current interpreter if uv is missing or unusable.
    """

    if preferred in {"auto", "uv"}:
        uv_runner = ensure_uv_runner(DEFAULT_UV_CACHE)
        if uv_runner is not None:
            return uv_runner
        if preferred == "uv":
            raise RuntimeError("uv executable not found on PATH")

    return [sys.executable, "-m", "pytest"], os.environ.copy()


def build_pytest_command(
    base_cmd: Sequence[str],
    options: TestOptions,
) -> list[str]:
    """Compose the final pytest command from the base runner and CLI flags."""

    cmd = list(base_cmd)

    marker = options.marker_expression()
    if marker:
        cmd.extend(["-m", marker])

    if options.coverage:
        cmd.extend(["--cov=src", "--cov-report=term-missing", "--cov-report=html"])

    if options.parallel:
        cmd.extend(["-n", "auto"])

    if options.verbose:
        cmd.append("-v")

    if options.failed:
        cmd.append("--lf")

    if options.junit_xml:
        cmd.extend(["--junitxml", str(options.junit_xml)])

    if options.pytest_args:
        cmd.extend(options.pytest_args)

    return cmd


def run_command(
    cmd: Sequence[str], description: str, env: Mapping[str, str] | None = None
) -> bool:
    """Execute a shell command and report whether it succeeded."""

    print(f"\n{'=' * 60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'=' * 60}")

    try:
        subprocess.run(cmd, check=True, env=dict(env or os.environ))
    except subprocess.CalledProcessError as error:
        print(f"\nError: {description} failed with return code {error.returncode}")
        return False
    except FileNotFoundError as error:
        print(f"\nError: Unable to run command ({error})")
        return False

    print(f"\nSuccess: {description} completed successfully")
    return True


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point used for both CLI execution and unit testing."""

    args = parse_arguments(argv)
    extra_pytest_args = normalize_pytest_args(args.pytest_args)

    options = TestOptions(
        test_type=args.type,
        coverage=args.coverage,
        parallel=args.parallel,
        verbose=args.verbose,
        failed=args.failed,
        junit_xml=args.junit,
        pytest_args=extra_pytest_args,
    )

    try:
        base_cmd, env = resolve_base_command(args.runner)
    except RuntimeError as error:
        print(f"\nError: Unable to prepare test environment: {error}")
        return 1

    pytest_cmd = build_pytest_command(base_cmd, options)
    if not run_command(pytest_cmd, f"Running {options.test_type} tests", env=env):
        return 1

    print(f"\nAll {options.test_type} tests completed successfully!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
