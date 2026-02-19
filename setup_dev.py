#!/usr/bin/env python3
"""
Development environment setup script for JSSP GNN project.

This script sets up the development environment with all necessary tools using uv.
"""

import shutil
import subprocess
import sys


def check_uv_installed():
    """Check if uv is installed."""
    if shutil.which("uv") is None:
        print("Error: uv is not installed. Please install uv first:")
        print("   Windows: winget install astral-sh.uv")
        print("   macOS/Linux: curl -LsSf https://astral.sh/uv/install.sh | sh")
        print("   Or visit: https://docs.astral.sh/uv/getting-started/installation/")
        sys.exit(1)


def run_command(cmd, description, check=True):
    """Run a command and handle errors."""
    print(f"\n[Action] {description}")
    print(f"Running: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, check=check, capture_output=True, text=True)
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(result.stderr)
        print(f"Success: {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error: {description} failed: {e}")
        if e.stdout:
            print(f"STDOUT: {e.stdout}")
        if e.stderr:
            print(f"STDERR: {e.stderr}")
        return False


def main():
    """Main setup function."""
    print("Setting up JSSP GNN development environment with uv...")

    # Check if uv is installed
    check_uv_installed()

    # Sync dependencies (creates venv and installs default + dev dependency groups)
    success = run_command(
        ["uv", "sync", "--dev"],
        "Syncing dependencies with uv (creating venv and installing default + dev groups)",
    )

    if not success:
        print("Error: Failed to sync dependencies. Please check your setup.")
        return

    # Install pre-commit hooks (package already managed by dev dependency group)
    run_command(
        ["uv", "run", "pre-commit", "install"],
        "Installing pre-commit hooks",
        check=False,
    )

    # Run initial formatting
    print("\nRunning initial code formatting...")
    run_command(
        [
            "uv",
            "run",
            "ruff",
            "check",
            "src",
            "tests",
            "--select",
            "I",
            "--fix",
        ],
        "Sorting imports with ruff",
        check=False,
    )

    run_command(
        ["uv", "run", "ruff", "format", "src", "tests"],
        "Formatting code with ruff",
        check=False,
    )

    # Run linting to check setup
    print("\nRunning initial code quality checks...")
    run_command(
        ["uv", "run", "ruff", "check", "src", "tests"],
        "Running ruff linting",
        check=False,
    )

    # Run type checking
    # run_command(
    #     ["uv", "run", "mypy", "src", "--ignore-missing-imports"],
    #     "Running mypy type checking",
    #     check=False,
    # )

    # Run tests to verify setup
    print("\nRunning tests to verify setup...")
    run_command(
        ["uv", "run", "pytest", "tests/unit", "-v"],
        "Running unit tests",
        check=False,
    )

    print("\nDevelopment environment setup complete!")
    print("\nNext steps:")
    print("1. Install recommended VS Code extensions (see .vscode/extensions.json)")
    print("2. Restart VS Code to apply the new settings")
    print("3. Start coding with automatic formatting and linting!")
    print("\nUseful commands:")
    print("  uv run make format      - Format code with ruff")
    print("  uv run make lint        - Run linting checks")
    print("  uv run make test        - Run all tests")
    print("  uv run make all-checks  - Run all quality checks")
    print("  uv sync --dev           - Sync dependencies (including dev group)")
    print("  uv add <package>        - Add a new dependency")
    print("  uv add --dev <package>  - Add a new dev dependency")


if __name__ == "__main__":
    main()
