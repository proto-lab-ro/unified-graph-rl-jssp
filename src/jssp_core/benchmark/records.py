from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class SolverType(StrEnum):
    """Enumeration of available solver types"""

    HEURISTIC = "heuristic"
    OPTIMAL = "optimal"
    GNN = "gnn"
    OTHER = "other"
    ML = "ml"


@dataclass(frozen=True)
class BenchmarkRecord:
    run_id: str  # Unique identifier for each created solver run
    benchmark_run_id: str
    instance_id: str
    solver_id: str
    solver_type: SolverType
    instance_hash: str

    instance_size: tuple[int, int]  # (num_jobs, num_machines)
    makespan: int
    computation_time_seconds: float

    solution: list[int] | None  # List of job indices in the order they were scheduled
    additional_metrics: dict[str, Any] = field(
        default_factory=dict
    )  # Use with caution for large data!
    error: str | None = None  # Error message if the solver failed


@dataclass(frozen=True)
class RewardRecord:
    run_id: str
    reward: float


@dataclass(frozen=True)
class BenchmarkRunRecord:
    benchmark_run_id: str  # Unique id for each start of a benchmark run
    timestamp: str
    seed: int
    generator_name: str
    generator_params: dict[str, Any]


@dataclass(frozen=True)
class SolverConfigRecord:
    solver_id: str
    solver_type: SolverType
    config_params: dict[str, Any]
