# Technical Guide: Unified Graph Framework for JSSP

This guide provides a concise technical overview of the unified-graph-rl-jssp repository, designed for researchers and developers to understand the project structure, setup, training, and inference.

## Architecture Overview

The codebase is split into two primary layers:
- **`jssp_core`**: A framework-agnostic foundation for the Job Shop Scheduling Problem (JSSP). It includes Gymnasium environments, observation/reward registries, and heuristic solvers.
- **`jssp_gnn`**: A deep learning layer built on TorchRL that implements the **Unified Graph Framework**. It includes GNN architectures, training loops, and inference utilities.

### Key Components
- **Environments**: Standard JSSP environments.
- **Observations**: **Feature-Based Homogenization** (e.g., `lb_bipartite_gnn`) maps heterogeneous JSSP states into a unified latent space for homogeneous GNN processing.
- **Rewards**: Flexible `RewardFunction` registry (e.g., `makespan_improvement`).
- **Solvers**: Unified interface (`JSSPSolverBase`) for heuristics and GNN policies.

## Setup

The project uses `uv` for dependency management.

```bash
# Clone and install in development mode
git clone https://github.com/proto-lab-ro/unified-graph-rl-jssp.git
cd unified-graph-rl-jssp
uv run setup_dev.py
```

`setup_dev.py` creates a virtual environment, syncs dependencies, and installs pre-commit hooks.

## Training GNN Policies

Training uses `src/jssp_gnn/train_gnn_matrix_form.py`. To achieve scale-invariant scheduling logic, we recommend training at the **structural saturation point** ($\mathcal{J} \approx \mathcal{M}$). The following command replicates the best-performing model described in our research.

### Basic Training (Structural Saturation Point)
```bash
uv run python src/jssp_gnn/train_gnn_matrix_form.py -m \
  --config-name=base_training experiment=empty \
  env.instance=20x20 \
  env.observation_provider=lb_bipartite_non_overlapping_gnn \
  training.total_frames=64000000 \
  training.frames_per_batch=1600 \
  training.num_epochs=1 \
  training.lr=0.0003
```

### Key Configuration Overrides
- `--config-name`: The base configuration file (required).
- `env.instance`: Path to a `.txt`/.jsp instance or a size string (e.g., "10x10").
- `env.observation_provider`: The GNN-compatible observation (e.g., `lb_bipartite_gnn`).
- `training.total_frames`: Total environment steps.
- `training.lr`: Learning rate.

Checkpoints and configurations are saved to `outputs/<date>/<time>/checkpoints/`.

## Inference & Evaluation

### Running a Trained Solver
The `GnnMatrixSolver` provides a high-level interface to run inference. It can load from a training checkpoint or a packaged model.

```python
from jssp_core.instances import get_instance
from jssp_gnn.solver import GnnMatrixSolver

# Load from a packaged model (.tar.gz)
solver = GnnMatrixSolver.from_package("model_store/my_model.tar.gz")

# Solve an instance
instance = get_instance("ft10")
schedule = solver.solve(instance)
print(f"Makespan: {schedule.get_makespan()}")
```

### Benchmarking
Compare GNN solvers against heuristics and optimal baselines using the comparison CLI:

```bash
python -m jssp_core.benchmark.solver_comparison_cli
  --dataset-path jssp_instances/ft06
  --experiment-root outputs/2026-02-18/12-00-00
  --job-heuristic spt --job-heuristic lpt
```

## Reproducibility

Use `jssp_core.reproducibility.set_seed(seed)` at the start of scripts to ensure deterministic behavior across `random`, `numpy`, and `torch`.

```python
from jssp_core import set_seed
set_seed(42)
```

## Examples

See the `examples/` directory for minimal usage scripts:
- `examples/inference.py`: Simple script to load and run a GNN solver.
- `examples/run_benchmark.py`: Comprehensive benchmarking script for comparing multiple solvers on various instances.
- `examples/plots.py`: Scripts to generate visualizations and plots from benchmark results (used in the academic paper).

For detailed architecture, refer to the self-documenting code in `src/`.
