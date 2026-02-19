# Makefile for JSSP GNN project testing and development

.PHONY: help install sync test test-unit test-integration test-fast test-slow test-gpu test-coverage clean lint format format-imports type-check all-checks

# Default target
help:
	@echo "Available targets:"
	@echo "  install         - Install the package and development dependencies"
	@echo "  sync            - Sync dependencies using uv"
	@echo "  test            - Run all tests"
	@echo "  test-unit       - Run unit tests only"
	@echo "  test-integration- Run integration tests only"
	@echo "  test-fast       - Run fast tests only (exclude slow tests)"
	@echo "  test-slow       - Run slow tests only"
	@echo "  test-gpu        - Run GPU tests only"
	@echo "  test-coverage   - Run tests with coverage reporting"
	@echo "  lint            - Run linting (ruff check)"
	@echo "  format          - Format code with ruff"
	@echo "  format-imports  - Auto-sort imports with ruff"
	@echo "  type-check      - Run type checking (mypy)"
	@echo "  all-checks      - Run all quality checks (lint, format, type-check)"
	@echo "  clean           - Clean up generated files"

# Installation
install:
	uv pip install -e ".[dev]"

sync:
	uv sync

# Testing targets
test:
	uv run pytest

test-unit:
	uv run pytest tests/unit -v

test-core:
	uv run pytest tests/unit/core -v

test-integration:
	uv run pytest tests/integration -v

test-fast:
	uv run pytest -m "not slow" -v

test-slow:
	uv run pytest -m "slow" -v

test-gpu:
	uv run pytest -m "gpu" -v

test-coverage:
	uv run pytest --cov=src --cov-report=term-missing --cov-report=html

test-parallel:
	uv run pytest -n auto

# Code quality targets
lint:
	uv run ruff check src tests

format:
	uv run ruff format src tests

format-imports:
	uv run ruff check src tests --select I --fix

format-check:
	uv run ruff format src tests --check
	uv run ruff check src tests --select I

type-check:
	uv run mypy src --ignore-missing-imports

# Combined quality checks
all-checks: lint format-check type-check

# Development workflow
dev-test: format lint test-unit

# Cleanup
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/
	rm -rf dist/
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf coverage.xml
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/

# Quick development feedback loop
quick: format lint test-unit
	@echo "Quick development checks completed!"

# Pre-commit hook simulation
pre-commit: format-imports format lint
	@echo "✅ Pre-commit checks completed!"
