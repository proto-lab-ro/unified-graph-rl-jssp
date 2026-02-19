"""
base.py
"""

import hashlib
import json
import tarfile
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, StrEnum
from pathlib import Path
from typing import Any, Protocol

from jssp_core.instances import JSSPInstance
from jssp_core.schedule import Schedule


@dataclass(frozen=True)
class SolveOutput:
    solution: Schedule
    info: dict[str, Any] = field(
        default_factory=dict
    )  # extras: rollout, returns, logs...


class EntityType(StrEnum):
    AGV = "agv"
    JOB = "job"
    OP = "operation"


class SolverProtocol(Protocol):
    def solve(self, instance: JSSPInstance) -> Schedule: ...

    def step(self, current_schedule: Schedule, *args, **kwargs) -> int: ...

    def get_name(self) -> str: ...

    def get_config_hash(self) -> str: ...

    def get_type(self) -> "SolverType": ...

    def solve_with_info(self, instance: JSSPInstance) -> "SolveOutput": ...


class Heuristic(ABC):
    def solve(self, instance: JSSPInstance) -> Schedule:
        """
        Solve gets an instance and returns a full solution.

        Default implementation creates a Schedule and calls step() until complete.
        Job-based heuristics can use this implementation directly.
        AGV-based heuristics should override this method.

        Args:
            instance: The JSSP instance to solve

        Returns:
            A complete Schedule object
        """
        schedule = Schedule(instance)
        while not schedule.is_complete():
            job_id = self.step(schedule)
            schedule.schedule_job(job_id)
        return schedule

    @abstractmethod
    def step(self, current_schedule: Schedule, **kwargs) -> int:
        """Returns the next Entity ID <EntityType> (eg. Job ID, AGV ID) to schedule, for MDP step"""
        pass

    def get_config_hash(self) -> str:
        """Return a stable identifier for heuristics that do not carry configs."""

        return self.__class__.__name__

    def __repr__(self) -> str:
        """Return a string representation of the heuristic."""
        return f"{self.__class__.__name__}()"

    def __str__(self) -> str:
        """Return the name of the heuristic."""
        return self.__class__.__name__

    def get_name(self) -> str:
        """Return the name of the heuristic."""
        return self.__class__.__name__

    @property
    def name(self) -> str:
        return self.get_name()

    def get_type(self) -> "SolverType":
        return SolverType.HEURISTIC

    def solve_with_info(self, instance: JSSPInstance) -> "SolveOutput":
        import time

        start_time = time.perf_counter()
        schedule = self.solve(instance)
        # end_time = time.perf_counter()
        # computation_time = end_time - start_time

        return SolveOutput(solution=schedule, info={})


class SolverType(Enum):
    """Enumeration of solver types for categorization and benchmarking."""

    HEURISTIC = "heuristic"  # Priority rules, dispatching rules
    OPTIMAL = "optimal"  # Constraint programming, exact methods
    ML = "ml"  # Machine learning (GNN, MLP, RL-based)
    HYBRID = "hybrid"  # Combination of multiple approaches
    METAHEURISTIC = "metaheuristic"  # GA, SA, Tabu Search, etc.


class JSSPSolverBase(ABC):
    """
    Abstract base class for all JSSP solvers.

    This provides a unified interface for solving Job Shop Scheduling Problems,
    enabling polymorphism for benchmarking and comparison across different
    solver types (heuristic, optimal, ML-based, etc.).

    Subclasses must implement:
        - solve(instance) -> solution

    Optionally override:
        - get_name() -> custom name
        - get_type() -> solver type classification
        - get_info() -> detailed metadata
        - solve_batch() -> optimized batch solving

    Example:
        class MySolver(JSSPSolverBase):
            def solve(self, instance):
                # Your solving logic here
                return solution

            def get_type(self):
                return SolverType.HEURISTIC
    """

    def __init__(self):
        """Initialize the solver base class."""
        self.cfg = None
        self._config_hash: str | None = None
        self.package_root: Path | None = None
        self.package_manifest: dict[str, Any] | None = None
        self.manifest_path: Path | None = None
        self._package_tmpdir: tempfile.TemporaryDirectory | None = None

    @classmethod
    def from_package(
        cls,
        package_path: str | Path,
        **solver_kwargs: Any,
    ):
        """
        Restore a solver from a packaged archive or extracted folder.

        Subclasses must implement `_create_from_package_root` to describe how a solver
        should be instantiated once the package contents are available.
        """
        package_root, tmp_dir = cls._prepare_package_root(Path(package_path))
        manifest = cls._load_manifest(package_root)

        solver = cls._create_from_package_root(
            package_root=package_root,
            manifest=manifest,
            **solver_kwargs,
        )
        solver.package_root = package_root
        solver.package_manifest = manifest
        solver.manifest_path = (
            package_root / "manifest.json"
            if (package_root / "manifest.json").exists()
            else None
        )
        solver._package_tmpdir = tmp_dir
        return solver

    @classmethod
    @abstractmethod
    def _create_from_package_root(
        cls,
        package_root: Path,
        manifest: dict[str, Any] | None,
        **solver_kwargs: Any,
    ):
        """
        Subclasses must override this hook to construct a solver using the extracted
        package directory. The method should return an initialized solver instance.
        """
        raise NotImplementedError

    @staticmethod
    def _load_manifest(package_root: Path) -> dict[str, Any] | None:
        manifest_path = package_root / "manifest.json"
        if manifest_path.exists():
            return json.loads(manifest_path.read_text(encoding="utf-8"))
        return None

    @staticmethod
    def _prepare_package_root(
        package_path: Path,
    ) -> tuple[Path, tempfile.TemporaryDirectory | None]:
        """Extract a package if needed and return the usable root path."""
        path = package_path.expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Package not found: {path}")

        if path.is_dir():
            return path, None

        tmp_dir = tempfile.TemporaryDirectory(prefix="solver_pkg_")
        extracted_root = Path(tmp_dir.name)
        with tarfile.open(path, "r:*") as tar:
            JSSPSolverBase._safe_extract(tar, extracted_root)

        subdirs = [p for p in extracted_root.iterdir() if p.is_dir()]
        root = subdirs[0] if len(subdirs) == 1 else extracted_root
        return root, tmp_dir

    @staticmethod
    def _safe_extract(tar: tarfile.TarFile, destination: Path) -> None:
        """Extract tar contents while preventing path traversal."""
        destination = destination.resolve()
        for member in tar.getmembers():
            member_path = destination / member.name
            if not str(member_path.resolve()).startswith(str(destination)):
                raise RuntimeError(f"Unsafe path detected in archive: {member.name}")
        tar.extractall(destination)

    def load_config_from_yaml(
        self,
        config_path: str | Path,
        config_name: str | None = None,
    ) -> None:
        """
        Load a Hydra configuration from a YAML file into self.cfg.

        This method supports two modes:
        1. Direct file loading: Load a specific YAML file directly
        2. Hydra compose: Use Hydra's compose system with a config directory and name

        Args:
            config_path: Path to the config directory (for Hydra compose) or
                        path to a specific YAML file (for direct loading).
            config_name: Name of the config file (without .yaml extension) when
                        using Hydra compose. If None, will load config_path as
                        a direct YAML file.

        Examples:
            # Direct YAML file loading
            solver.load_config_from_yaml("path/to/config.yaml")

            # Hydra compose loading
            solver.load_config_from_yaml(
                config_path="conf/gnn",
                config_name="config"
            )

        Raises:
            FileNotFoundError: If the config file doesn't exist.
            ValueError: If the config format is invalid.
        """
        from omegaconf import OmegaConf

        config_path = Path(config_path)

        if config_name is None:
            # Direct YAML file loading
            if not config_path.exists():
                raise FileNotFoundError(f"Config file not found: {config_path}")

            self.cfg = OmegaConf.load(config_path)
            self._config_hash = self._compute_config_hash(
                self._config_to_dict(resolve=False)
            )
        else:
            # Hydra compose loading
            from hydra import compose, initialize
            from hydra.core.global_hydra import GlobalHydra

            if not config_path.exists():
                raise FileNotFoundError(f"Config directory not found: {config_path}")

            # Clear any existing Hydra instance
            GlobalHydra.instance().clear()

            # Initialize Hydra with the config path
            with initialize(config_path=str(config_path), version_base=None):
                self.cfg = compose(config_name=config_name)
                self._config_hash = self._compute_config_hash(
                    self._config_to_dict(resolve=False)
                )

    @abstractmethod
    def solve(self, instance: JSSPInstance) -> Schedule:
        """
        Solve a single JSSP instance and return a solution.

        Args:
            instance: The JSSP instance to solve, represented as a list of jobs
                     where each job is a list of (machine_id, duration) tuples.

        Returns:
            A Schedule object containing the schedule, makespan, and other
            solution information.

        Raises:
            NotImplementedError: If the subclass hasn't implemented this method.
            RuntimeError: If solving fails or times out.
        """
        pass

    def solve_with_info(self, instance: "JSSPInstance") -> SolveOutput:
        """
        Optional richer API. Default falls back to solve() and no extras.
        Subclasses (e.g., RL) can override to return rollout/returns/etc.
        """
        sol = self.solve(instance)
        return SolveOutput(solution=sol, info={})

    def get_name(self) -> str:
        """
        Return a human-readable name for this solver.

        Default implementation returns the class name.
        Override to provide a more descriptive name.

        Returns:
            String identifier for the solver (e.g., "SPT Heuristic", "GNN Policy")
        """
        return self.__class__.__name__

    def get_type(self) -> SolverType:
        """
        Return the type/category of this solver.

        Used for grouping solvers in benchmarks and analysis.
        Default is HEURISTIC.

        Returns:
            A SolverType enum value indicating the solver category.
        """
        return SolverType.HEURISTIC

    @property
    def name(self) -> str:
        return self.get_name()

    def get_info(self) -> dict[str, Any]:
        """
        Return detailed metadata about this solver.

        Useful for logging, benchmarking, and reproducibility.
        Override to include solver-specific configuration.

        Returns:
            Dictionary with solver information. Default keys:
                - name: Solver name
                - type: Solver type (heuristic/optimal/ml/etc.)

        Example:
            {
                "name": "GNN PPO Solver",
                "type": "ml",
                "model_path": "/path/to/model.pt",
                "device": "cuda",
                "max_steps": 500,
            }
        """
        return {
            "name": self.get_name(),
            "type": self.get_type(),
        }

    def solve_batch(
        self, instances: list[JSSPInstance], show_progress: bool = False
    ) -> list[Schedule]:
        """
        Solve multiple instances.

        Default implementation solves sequentially.
        Override in subclasses to provide optimized batch processing
        (e.g., parallel solving, GPU batching for ML models).

        Args:
            instances: List of JSSP instances to solve.
            show_progress: Whether to display progress (e.g., with tqdm).

        Returns:
            List of Schedule objects, one per instance.

        Example:
            solver = MySolver()
            instances = [instance1, instance2, instance3]
            solutions = solver.solve_batch(instances)
        """
        if show_progress:
            try:
                from tqdm import tqdm

                return [self.solve(instance) for instance in tqdm(instances)]
            except ImportError:
                pass  # Fall through to non-progress version

        return [self.solve(instance) for instance in instances]

    def benchmark(self, instance: JSSPInstance, num_runs: int = 1) -> dict[str, Any]:
        """
        Benchmark this solver on an instance with timing information.

        Args:
            instance: The instance to solve
            num_runs: Number of times to run (for averaging)

        Returns:
            Dictionary with benchmark results:
                - makespan: Best makespan found
                - avg_time: Average solving time (seconds)
                - num_runs: Number of runs performed
        """
        import time

        times = []
        best_solution = None
        best_makespan = float("inf")

        for _ in range(num_runs):
            start = time.time()
            solution = self.solve(instance)
            elapsed = time.time() - start

            times.append(elapsed)
            if solution.makespan < best_makespan:
                best_makespan = solution.makespan
                best_solution = solution

        return {
            "solver": self.get_name(),
            "makespan": best_makespan,
            "avg_time": sum(times) / len(times),
            "min_time": min(times),
            "max_time": max(times),
            "num_runs": num_runs,
        }

    def __repr__(self) -> str:
        """String representation for debugging."""
        return f"{self.get_name()}(type={self.get_type().value})"

    def __str__(self) -> str:
        """Human-readable string representation."""
        return self.get_name()

    # --- Configuration helpers ---
    def _config_to_dict(self, resolve: bool = False) -> dict[str, Any]:
        """Return cfg as a plain dict without mutating the original object."""

        cfg = getattr(self, "cfg", None)
        if cfg is None:
            return {}
        if isinstance(cfg, dict):
            return cfg

        try:
            from omegaconf import OmegaConf

            container = OmegaConf.to_container(cfg, resolve=resolve)
        except Exception:
            container = None
        if isinstance(container, dict):
            return container
        return {}

    @staticmethod
    def _compute_config_hash(config: dict[str, Any]) -> str:
        """Generate a deterministic hash for a config dictionary."""

        if not config:
            return "none"
        payload = json.dumps(config, sort_keys=True, default=str)
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]

    def refresh_config_hash(self) -> str:
        """Recompute the config hash from the current cfg state."""

        self._config_hash = self._compute_config_hash(self._config_to_dict())
        return self._config_hash

    def get_config_hash(self) -> str:
        """Expose the cached config hash for benchmarking/logging."""

        if not self._config_hash:
            return self.refresh_config_hash()
        return self._config_hash

    def set_config_hash(self, value: str | None) -> None:
        """Allow overriding the stored config hash (e.g., when loaded externally)."""

        self._config_hash = value


# Export for convenience
__all__ = ["JSSPSolverBase", "SolverType"]
