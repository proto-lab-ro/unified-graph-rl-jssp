# Scalable Production Scheduling: Linear Complexity via Unified Homogeneous Graphs

This repository implements the **Unified Graph Framework** for the Job Shop Scheduling Problem (JSSP), as presented in our research. It provides a scalable Reinforcement Learning (RL) system that leverages linear-complexity graph representations and feature-based homogenization to achieve robust zero-shot generalization across varying problem scales.

## Overview

The system is built on two primary layers:
- **`jssp_core`**: A framework-agnostic foundation containing JSSP environments, observation/reward registries, and heuristic baselines.
- **`jssp_gnn`**: The deep learning layer utilizing TorchRL and PyTorch Geometric to train PPO agents with a homogeneous GIN backbone on our unified graph representation.

### Key Contributions
- **Linear-Complexity Unified Graph**: A sparse heterogeneous graph representation that models machines as first-class entities, reducing topological complexity from quadratic to linear.
- **Feature-Based Homogenization**: A strategy to project distinct node roles (operations and machines) into a shared latent space, enabling processing via standard homogeneous GNNs.
- **Structural Saturation**: A training methodology focused on the "saturation point" ($\mathcal{J} \approx \mathcal{M}$) to induce scale-invariant scheduling logic.

## Quick Start

### Installation
We recommend using [uv](https://github.com/astral-sh/uv) for execution and dependency management.

```bash
uv run setup_dev.py
```

### Inference Example
Load a pre-trained model and solve a JSSP instance (e.g., `ft06`).

```python
from jssp_core.instances import get_instance
from jssp_gnn.solver import GnnMatrixSolver

# Load the solver from a packaged model
solver = GnnMatrixSolver.from_package("examples/20x20_model.tar.gz")

# Solve a standard benchmark instance
instance = get_instance("ft06")
schedule = solver.solve(instance)
print(f"Makespan: {schedule.get_makespan()}")
```

### Training Example
Train a policy using the structural saturation configuration (20x20), which induces scale-invariant logic for zero-shot generalization as described in the paper.

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

## Documentation

For technical details on the unified graph architecture, training regimes, and zero-shot evaluation, please see the **[Technical Guide](docs/TECHNICAL_GUIDE.md)**.

## Examples

Usage scripts are located in the `examples/` directory:
- `inference.py`: Minimal script to load and run a GNN solver.
- `run_benchmark.py`: Benchmarking multiple solvers on various instances.
- `plots.py`: Generating visualizations and plots from benchmark results (as seen in the paper).

## Testing & Quality

```bash
make test          # Run all tests
make all-checks    # Linting, formatting, and type-checking
```

---
Happy scheduling!
