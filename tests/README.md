# Testing Guide

The repository now follows a tidy, layered testing layout so you can immediately find the scope of any check and pick the right entry-point for local or CI runs.

## Directory Layout

```
tests/
├── conftest.py              # project wide fixtures and hooks
├── integration/             # cross-component or end-to-end flows
│   └── test_training_integration.py
├── unit/
│   ├── domain/              # core scheduling/domain logic
│   │   ├── test_heuristics.py
│   │   ├── test_instances.py
│   │   ├── test_optimal_solver.py
│   │   ├── test_schedule.py
│   │   └── test_TestGetOperationsOnMachine.py
│   ├── env/                 # environment + reward/observation utilities
│   │   ├── test_environment.py
│   │   ├── test_observation_providers.py
│   │   └── test_reward_functions.py
│   └── services/            # IO, packaging, syncing, persistence helpers
│       ├── test_connection.py
│       ├── test_model_packager.py
│       ├── test_model_sync.py
│       ├── test_solution_dataset_manager.py
│       └── test_storage_backends.py
└── README.md                # this guide
```

The category folders match the way the production code is divided (core domain, RL environments, and service/infra helpers). When adding a new file, drop it in the folder that mirrors the production module you are testing.

## Markers and suites

Common pytest markers supported across the suite:

| Marker | Description |
| --- | --- |
| `unit` | Small, deterministic checks that isolate one class or helper |
| `integration` | Multi-component flows such as full training loops |
| `slow` | Expensive, long running tests |
| `fast` | Convenience alias provided by `run_tests.py` (`not slow`) |
| `gpu` | GPU required – auto-skipped when CUDA is unavailable |
| `env`, `model`, `training` | More granular tags for filtering domain specific logic |

Use markers when you add tests so that targeted runs (`pytest -m unit`, `python run_tests.py --type fast`, etc.) stay meaningful.

## Running tests

Choose the driver that best fits your workflow:

### `run_tests.py`

```bash
python run_tests.py              # all tests with default markers
python run_tests.py --type unit  # only the unit suite
python run_tests.py --type fast  # skip @slow tests
python run_tests.py --coverage   # include coverage reports
python run_tests.py --parallel   # enable pytest-xdist
python run_tests.py -- --maxfail=1 -k Env
```

The script auto-detects `uv run -m pytest` when available and falls back to the active interpreter otherwise.

### Plain pytest

```bash
pytest                     # full suite with strict defaults
pytest tests/unit/env      # run a folder
pytest -m "unit and not gpu" -k reward
pytest tests/unit/domain/test_schedule.py::TestSchedule::test_reset
```

### Make targets

```bash
make test             # all tests
make test-unit        # unit suite
make test-integration # integration suite
make test-fast        # skip slow
make test-coverage    # add coverage reports
```

## Fixtures

`tests/conftest.py` holds project-wide fixtures. Highlights:

- Device and deterministic RNG (`device`, `set_random_seed`).
- Canonical instances/schedules (FT06, 3x3) and ready-made environments.
- Hydra / SB3 config helpers (`hydra_gnn_config`, `hydra_sb3_config`, `create_temp_hydra_config`).
- Utility fixtures such as `temp_model_path`, `sample_config`, and `mock_tensorboard_writer`.

Prefer reusing these fixtures instead of duplicating setup logic—keeping setup centralized is what allows the suite to remain short and readable.

## Configuration & coverage

All pytest settings now live in `pyproject.toml` under `[tool.pytest.ini_options]`, so there is a single source of truth for discovery rules, default CLI flags, and markers. Coverage is enabled by default (`--cov=src`, `--cov-report=term-missing`, `--cov-report=html:htmlcov`, `--cov-report=xml`). HTML output is placed under `htmlcov/` and XML under `coverage.xml` for CI to pick up.

## Continuous integration

GitHub Actions runs the same commands defined above: lint/format/type-check via the Makefile and `python run_tests.py --type all --coverage`. When you update or add tests, make sure any new assets live under `tests/` (tracked automatically via `testpaths`) and update this guide if you introduce a new category or workflow.

## Best practices

1. Keep unit tests fast—mock remote resources and skip GPU-heavy paths unless explicitly marked.
2. Share fixtures or helper factories; if a setup is used by more than one module, promote it to `conftest.py`.
3. Name tests after the behavior they assert (`test_schedule_handles_negative_durations`).
4. Prefer parametrization over copy/pasting similar tests.
5. Aim for meaningful coverage, not just numbers—tests should describe behavior in terms of domain concepts.
