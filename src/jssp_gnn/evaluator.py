"""
Modular evaluation system for training.

This module provides a flexible evaluation framework with the following components:

1. BaseEvaluator (Abstract Base Class):
   - Defines the interface for all evaluators
   - Includes should_evaluate() for frequency control

2. StandardEvaluator:
   - Performs single-instance deterministic rollouts
   - Logs metrics: reward, return, makespan, gap to lower bound
   - Tracks best results and saves best model (optional)
   - Supports customizable metric tracking (min or max)
   - Default evaluator for single-instance training

3. BenchmarkEvaluator:
   - Evaluates policy across multiple instances from an InstanceGenerator
   - Supports dataset files, random generators, and heuristic-based references
   - Compares against known best heuristic makespans (lower bounds) when available
   - Logs aggregate metrics: mean/median/min/max gaps and makespans
   - Useful for comprehensive generalization testing

4. Instance Generator Functions:
   - Uses InstanceGenerator from jssp_core.instances.generators
   - simple_instance_generator: Wraps InstanceGenerator with optional fixed reference makespan
   - heuristic_instance_generator: Computes reference makespan using a heuristic solver

5. NoOpEvaluator:
   - Disables evaluation (for debugging or fast training)

6. create_evaluator() Factory:
   - Creates evaluator from configuration
   - Enables easy switching between evaluation strategies

Configuration Example (in YAML):
    evaluation:
      type: "standard"  # or "benchmark" or "none"
      max_steps: 1000
      lower_bound: 55  # optional, for gap calculation (standard only)
      save_best_model: true  # optional, default: false (standard only)
      save_dir: "best_model"  # required if save_best_model=true
      metric_mode: "min"  # "min" for makespan, "max" for reward
      metric_key: "makespan"  # metric to track for best model

      # For benchmark evaluator with dataset file:
      instance_size: [10, 10]  # size of instances to evaluate
      dataset_filename: "jssp_instances/heuristic_solutions/solved_instances.json"
      n_instances: 100  # number of instances to evaluate
      benchmark_heuristic: "mwkr"  # reference heuristic name for gaps

      # For benchmark evaluator with instance generator:
      use_generator: true
      generator_type: "random_uniform"  # or any registered generator
      generator_kwargs:
        min_duration: 1
        max_duration: 10
      n_instances: 100
      reference_makespan: 55  # optional fixed reference for gap calculation

      # For benchmark evaluator with on-the-fly heuristic reference:
      use_generator: true
      use_heuristic_reference: true
      generator_type: "random_uniform"
      generator_kwargs:
        min_duration: 1
        max_duration: 10
      heuristic_name: "mwr"  # heuristic to compute reference (spt, mwr, lpt, etc.)
      n_instances: 100

Usage in training loop:
    evaluator = create_evaluator(env, logger, cfg)
    if evaluator.should_evaluate(step, eval_freq):
        metrics = evaluator.evaluate(policy_module, step)

    # Get best results at end of training (for StandardEvaluator):
    if hasattr(evaluator, 'get_best_metric'):
        best_metric = evaluator.get_best_metric()
        best_step = evaluator.get_best_step()

Direct usage with custom generators:
    from jssp_gnn.evaluator import (
        BenchmarkEvaluator,
        dataset_instance_generator,
        simple_instance_generator,
        heuristic_instance_generator,
    )
    from jssp_core.instances.generators import RandomInstanceGenerator

    # Using dataset
    gen = dataset_instance_generator(
        instance_size=(10, 10),
        dataset_filename="path/to/dataset.json",
        benchmark_heuristic="mwkr",
    )

    # Or using generator with fixed reference
    generator = RandomInstanceGenerator(num_jobs=10, num_machines=10)
    gen = simple_instance_generator(generator=generator, reference_makespan=55)

    # Or using generator with on-the-fly heuristic reference
    generator = RandomInstanceGenerator(num_jobs=10, num_machines=10)
    gen = heuristic_instance_generator(generator=generator, heuristic_name="mwr")

    evaluator = BenchmarkEvaluator(env, logger, instance_generator=gen, n_instances=100)
"""

import csv
import hashlib
import os
from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass

import torch
from omegaconf import DictConfig
from torchrl.envs import TransformedEnv
from torchrl.envs.utils import ExplorationType, set_exploration_type

from jssp_core.domain.base_dispatcher import DispatcherBase
from jssp_core.instances.generators import InstanceGenerator
from jssp_core.instances.jssp import JSSPInstance
from jssp_core.solver.base import SolverProtocol
from jssp_gnn.logger import TrainingLogger


def _unwrap_base_env(env: TransformedEnv):
    """Walk through nested TransformedEnvs to get the underlying base env."""
    base_env = env
    while hasattr(base_env, "base_env"):
        base_env = base_env.base_env
    return base_env


@dataclass
class EvaluationInstance:
    """Container for an instance and its reference makespan (if available)."""

    instance: JSSPInstance
    reference_makespan: float | None = None
    metadata: dict | None = None


class BaseEvaluator(ABC):
    """Base class for evaluation strategies during training."""

    def __init__(self, env: TransformedEnv, logger: TrainingLogger):
        """
        Initialize the evaluator.

        Args:
            env: The environment to evaluate on
            logger: Logger for recording evaluation metrics
        """
        self.env = env
        self.logger = logger
        self._closed = False

    @abstractmethod
    def evaluate(self, policy_module, step: int) -> dict[str, float]:
        """
        Run evaluation and return metrics.

        Args:
            policy_module: The policy to evaluate
            step: Current training step

        Returns:
            Dictionary of evaluation metrics
        """
        ...

    def should_evaluate(self, step: int, eval_freq: int) -> bool:
        """
        Determine if evaluation should be run at this step.

        Args:
            step: Current training step
            eval_freq: Frequency of evaluation

        Returns:
            True if evaluation should be run
        """
        return step % eval_freq == 0 if eval_freq > 0 else False

    def close(self):
        """Clean up resources if needed."""
        if self._closed:
            return
        if self.env is not None:
            try:
                self.env.close()
            except Exception as e:
                print(f"[Evaluator] Warning: Failed to close env (close): {e}")

        self._closed = True


class NoOpEvaluator(BaseEvaluator):
    """Evaluator that does nothing - for disabling evaluation."""

    def evaluate(self, policy_module, step: int) -> dict[str, float]:
        """
        No-op evaluation.

        Args:
            policy_module: The policy to evaluate (unused)
            step: Current training step (unused)

        Returns:
            Empty dictionary
        """
        return {}


class StandardEvaluator(BaseEvaluator):
    """
    Standard evaluator that runs a single rollout and logs metrics.

    Tracks best results and optionally saves the best model.
    """

    def __init__(
        self,
        env: TransformedEnv,
        logger: TrainingLogger,
        max_steps: int = 1000,
        lower_bound: float | None = None,
        save_best_model: bool = False,
        save_dir: str | None = None,
        metric_mode: str = "min",  # "min" for makespan, "max" for reward
        metric_key: str = "makespan",
    ):
        """
        Initialize standard evaluator.

        Args:
            env: The environment to evaluate on
            logger: Logger for recording evaluation metrics
            max_steps: Maximum steps for evaluation rollout
            lower_bound: Known lower bound for gap calculation (optional)
            save_best_model: Whether to save the best model based on evaluation metric
            save_dir: Directory to save best model (required if save_best_model=True)
            metric_mode: "min" to minimize metric (e.g., makespan) or "max" to maximize (e.g., reward)
            metric_key: Key of the metric to track for best model (e.g., "makespan", "avg_return")
        """
        super().__init__(env, logger)
        self.max_steps = int(max_steps)
        self.lower_bound = lower_bound
        self.save_best_model = save_best_model
        self.save_dir = save_dir
        self.metric_mode = metric_mode
        self.metric_key = metric_key

        # Initialize best metric tracking
        if metric_mode == "min":
            self.best_metric = float("inf")
        else:
            self.best_metric = -float("inf")

        self.best_step: int | None = None

        # Validate save configuration
        if save_best_model and save_dir is None:
            raise ValueError("save_dir must be provided when save_best_model=True")

    def evaluate(self, policy_module, step: int) -> dict[str, float]:
        """
        Run a single deterministic rollout and log metrics.

        Tracks best results and saves the model if it achieves a new best metric.

        Args:
            policy_module: The policy to evaluate
            step: Current training step

        Returns:
            Dictionary of evaluation metrics
        """
        with set_exploration_type(ExplorationType.DETERMINISTIC), torch.no_grad():
            eval_rollout = self.env.rollout(
                self.max_steps,
                policy_module,
            )

            # Extract metrics from rollout
            eval_mask = eval_rollout["next", "done"]
            eval_terminated = eval_rollout["next", "terminated"]

            metrics = {
                "mean_reward": eval_rollout["next", "reward"].mean().item(),
                "sum_reward": eval_rollout["next", "reward"].sum().item(),
                "step_count": eval_rollout["step_count"].max().item(),
                "avg_return": eval_rollout["next", "episode_reward"][eval_mask]
                .mean()
                .item(),
                "makespan": eval_rollout["next", "makespan"][eval_terminated].item(),
            }

            # Log evaluation metrics
            self.logger.log_evaluation_metrics(
                mean_reward=metrics["mean_reward"],
                sum_reward=metrics["sum_reward"],
                step_count=metrics["step_count"],
                avg_return=metrics["avg_return"],
                makespan=metrics["makespan"],
                step=step,
            )

            # Calculate and log gap to lower bound if available
            if self.lower_bound is not None:
                gap_to_lb = (metrics["makespan"] - self.lower_bound) / self.lower_bound
                self.logger.log_custom_metrics({"eval/gap_to_lb": gap_to_lb}, step=step)
                metrics["gap_to_lb"] = gap_to_lb

            # Check if this is the best model and save if enabled
            if self.save_best_model:
                current_metric = metrics.get(self.metric_key)
                if current_metric is not None:
                    is_best = False
                    if self.metric_mode == "min":
                        is_best = current_metric < self.best_metric
                    else:
                        is_best = current_metric > self.best_metric

                    if is_best:
                        self.best_metric = current_metric
                        self.best_step = step
                        self._save_model(policy_module, step, current_metric)

                        # Log best metric
                        self.logger.log_custom_metrics(
                            {f"eval/best_{self.metric_key}": self.best_metric},
                            step=step,
                        )
                        print(
                            f"New best {self.metric_key}: {self.best_metric:.4f} at step {step}"
                        )

            del eval_rollout
            return metrics

    def _save_model(self, model, step: int, metric_value: float):
        """
        Save the model checkpoint.

        Args:
            model: The model to save
            step: Current training step
            metric_value: Value of the tracked metric
        """
        import os

        # save_dir is guaranteed to be not None due to validation in __init__
        assert self.save_dir is not None
        save_dir = self.save_dir  # Store in local variable for type checker

        # Create save directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)

        # Handle tuple models (for multi-component models)
        if isinstance(model, tuple):
            separate_models = model[0]
            main_model = model[1]

            # Save separate models
            for key, small_model in separate_models.items():
                checkpoint_path = os.path.join(save_dir, f"{key}_best_model.pt")
                torch.save(small_model.state_dict(), checkpoint_path)

            # Save main model
            main_checkpoint_path = os.path.join(save_dir, "best_model.pt")
            torch.save(main_model.state_dict(), main_checkpoint_path)
        else:
            # Save single model
            checkpoint_path = os.path.join(save_dir, "best_model.pt")
            torch.save(model.state_dict(), checkpoint_path)

        # Save metadata about the best model
        metadata_path = os.path.join(save_dir, "best_model_metadata.txt")
        with open(metadata_path, "w", encoding="utf-8") as f:
            f.write(f"Best {self.metric_key}: {metric_value:.6f}\n")
            f.write(f"Step: {step}\n")
            f.write(f"Mode: {self.metric_mode}\n")

    def get_best_metric(self) -> float:
        """
        Get the best metric value achieved during evaluation.

        Returns:
            Best metric value
        """
        return self.best_metric

    def get_best_step(self) -> int | None:
        """
        Get the step at which the best metric was achieved.

        Returns:
            Step number or None if no evaluation has been performed
        """
        return self.best_step


def simple_instance_generator(
    generator: InstanceGenerator,
    reference_makespan: float | None = None,
) -> Iterator[EvaluationInstance]:
    """
    Wrap an InstanceGenerator to yield EvaluationInstance objects with fixed reference.

    Args:
        generator: Instance generator to use
        reference_makespan: Optional fixed reference makespan for all instances

    Yields:
        EvaluationInstance with generated instance and optional fixed reference
    """
    for instance in generator:
        yield EvaluationInstance(
            instance=instance,
            reference_makespan=reference_makespan,
        )


def heuristic_instance_generator(
    generator: InstanceGenerator,
    heuristic_name: str = "mwr",
) -> Iterator[EvaluationInstance]:
    """
    Wrap an InstanceGenerator to compute reference makespans using a heuristic solver.

    Args:
        generator: Instance generator to use
        heuristic_name: Name of heuristic to use for reference makespan (e.g., "spt", "mwr", "lpt")

    Yields:
        EvaluationInstance with generated instance and heuristic-computed reference
    """
    from jssp_core.solver.heuristic_solver import JSSPHeuristicSolver

    heuristic_name = heuristic_name.lower()
    for instance in generator:
        # Solve with heuristic to get reference makespan
        heuristic_solver = JSSPHeuristicSolver(instance)
        if heuristic_name == "all":
            heuristic_results = heuristic_solver.solve_all_heuristics()
            best_heuristic, best_schedule = min(
                heuristic_results.items(), key=lambda item: item[1].get_makespan()
            )

            reference_makespan = best_schedule.get_makespan()

            yield EvaluationInstance(
                instance=instance,
                reference_makespan=reference_makespan,
                metadata={
                    "heuristic_name": best_heuristic,
                    "heuristic_makespan": reference_makespan,
                },
            )

        else:
            heuristic_schedule = heuristic_solver.solve_with_heuristic(heuristic_name)
            reference_makespan = heuristic_schedule.get_makespan()

            yield EvaluationInstance(
                instance=instance,
                reference_makespan=reference_makespan,
                metadata={
                    "heuristic_name": heuristic_name,
                    "heuristic_makespan": reference_makespan,
                },
            )


def create_test_env(dispatcher):
    _ = dispatcher.setup_instance()
    test_env = dispatcher.setup_test_environment()
    return test_env


class BenchmarkEvaluator(BaseEvaluator):
    """
    Evaluates a policy against multiple instances using an InstanceGenerator.

    Compares policy solutions against known best heuristic makespans (lower bounds) if available.
    """

    def __init__(
        self,
        env: TransformedEnv,
        logger: TrainingLogger,
        instance_generator: Iterator[EvaluationInstance],
        n_instances: int = 100,
        max_steps: int = 1000,
        save_best_model: bool = False,
        save_dir: str | None = None,
        metric_mode: str = "min",
        metric_key: str = "mean_gap",
    ):
        """
        Initialize benchmark evaluator.

        Args:
            env: The environment to evaluate on
            logger: Logger for recording evaluation metrics
            instance_generator: Generator that yields EvaluationInstance objects
            n_instances: Number of instances to evaluate
            max_steps: Maximum steps for evaluation rollout
            save_best_model: Whether to store the best-performing model checkpoint
            save_dir: Directory where the checkpoint is written
            metric_mode: "min" or "max" comparison strategy for metric_key
            metric_key: Aggregated metric from evaluate() to track
        """
        super().__init__(env, logger)

        self.instance_generator = instance_generator
        self.n_instances = n_instances
        self.max_steps = max_steps
        self.save_best_model = save_best_model
        self.save_dir = save_dir
        self.metric_mode = metric_mode
        self.metric_key = metric_key

        # Cache underlying GraphMatrixEnv to validate dimensions
        self._graph_env = _unwrap_base_env(env)
        self._env_num_jobs = getattr(self._graph_env, "num_jobs", None)
        self._env_num_machines = getattr(self._graph_env, "num_machines", None)

        if metric_mode not in {"min", "max"}:
            raise ValueError(f"metric_mode must be 'min' or 'max', got {metric_mode}")

        self.best_metric = float("inf") if metric_mode == "min" else -float("inf")
        self.best_step: int | None = None

        if save_best_model and save_dir is None:
            raise ValueError("save_dir must be provided when save_best_model=True")

    def evaluate(self, policy_module: SolverProtocol, step: int) -> dict[str, float]:
        """
        Run evaluation across multiple instances and compare against lower bounds.

        Args:
            policy_module: The policy solver to evaluate (must implement solve() method)
            step: Current training step

        Returns:
            Dictionary of evaluation metrics including gaps and makespans
        """
        gaps = []
        policy_makespans = []
        lower_bounds = []
        detailed_results = []

        # Iterate over instances from the generator
        worst_instances = []
        instance_count = 0
        for eval_instance in self.instance_generator:
            if instance_count >= self.n_instances:
                break
            instance_count += 1
            # Validate each instance to fail fast if dimensions mismatch
            if self._env_num_jobs is not None and self._env_num_machines is not None:
                inst_jobs = eval_instance.instance.num_jobs()
                inst_machines = eval_instance.instance.num_machines()
                if (
                    inst_jobs != self._env_num_jobs
                    or inst_machines != self._env_num_machines
                ):
                    raise ValueError(
                        "Benchmark instance dimensions "
                        f"({inst_jobs} jobs, {inst_machines} machines) do not match "
                        f"the training environment ({self._env_num_jobs} jobs, "
                        f"{self._env_num_machines} machines). "
                        "Ensure instance provider generates matching instances."
                    )

            # Load instance into environment and solve with policy
            with set_exploration_type(ExplorationType.DETERMINISTIC), torch.no_grad():
                # Set the new instance in the base environment
                self.env.env.instance = eval_instance.instance

                # Reset environment with new instance
                self.env.reset()

                # Run rollout
                eval_rollout = self.env.rollout(
                    self.max_steps,
                    policy_module,
                )

                # Extract makespan
                eval_terminated = eval_rollout["next", "terminated"]
                policy_makespan = eval_rollout["next", "makespan"][
                    eval_terminated
                ].item()

            # Calculate gap if reference makespan is available
            gap = None
            if eval_instance.reference_makespan is not None:
                gap = (
                    policy_makespan - eval_instance.reference_makespan
                ) / eval_instance.reference_makespan
                gaps.append(gap)
                lower_bounds.append(eval_instance.reference_makespan)

                if gap > 0.05:  # Track worst instances with gap > 5%
                    worst_instances.append(eval_instance.instance)

            policy_makespans.append(policy_makespan)

            # Store detailed result
            instance_name = f"instance_{instance_count}"
            if eval_instance.metadata and "dataset_result" in eval_instance.metadata:
                # Try to get a name from dataset result if available
                if hasattr(eval_instance.metadata["dataset_result"], "name"):
                    instance_name = eval_instance.metadata["dataset_result"].name

            # Calculate instance hash for tracking
            instance_str = str(eval_instance.instance)
            instance_hash = hashlib.md5(instance_str.encode()).hexdigest()[:8]

            detailed_results.append(
                {
                    "instance_name": instance_name,
                    "instance_hash": instance_hash,
                    "policy_makespan": policy_makespan,
                    "reference_makespan": eval_instance.reference_makespan,
                    "gap": gap,
                }
            )

        # Compute aggregate metrics
        metrics = {
            "mean_makespan": sum(policy_makespans) / len(policy_makespans)
            if len(policy_makespans) > 0
            else float("inf"),
            "num_instances": len(policy_makespans),
        }
        metrics["worst_instances"] = worst_instances

        # Add gap metrics only if reference makespans are available
        if gaps:
            metrics.update(
                {
                    "mean_gap": sum(gaps) / len(gaps),
                    "median_gap": sorted(gaps)[len(gaps) // 2],
                    "max_gap": max(gaps),
                    "min_gap": min(gaps),
                    "mean_lower_bound": sum(lower_bounds) / len(lower_bounds),
                }
            )

        # Log metrics
        log_dict = {"benchmark/mean_makespan": metrics["mean_makespan"]}
        if gaps:
            log_dict.update(
                {
                    "benchmark/mean_gap": metrics["mean_gap"],
                    "benchmark/median_gap": metrics["median_gap"],
                    "benchmark/max_gap": metrics["max_gap"],
                    "benchmark/min_gap": metrics["min_gap"],
                    "benchmark/mean_lower_bound": metrics["mean_lower_bound"],
                }
            )

        self.logger.log_custom_metrics(log_dict, step=step)

        print(f"\n=== Benchmark Evaluation at step {step} ===")
        print(f"Evaluated {metrics['num_instances']} instances")
        print(f"Mean makespan: {metrics['mean_makespan']:.2f}")

        if gaps:
            print(
                f"Mean gap to lower bound: {metrics['mean_gap']:.4f} ({metrics['mean_gap'] * 100:.2f}%)"
            )
            print(f"Median gap: {metrics['median_gap']:.4f}")
            print(f"Min/Max gap: {metrics['min_gap']:.4f} / {metrics['max_gap']:.4f}")
            print(f"Mean lower bound: {metrics['mean_lower_bound']:.2f}")

            # Calculate win/loss/tie
            better = sum(
                1 for r in detailed_results if r["gap"] is not None and r["gap"] < 0
            )
            equal = sum(
                1
                for r in detailed_results
                if r["gap"] is not None and abs(r["gap"]) < 1e-6
            )
            worse = sum(
                1 for r in detailed_results if r["gap"] is not None and r["gap"] > 1e-6
            )

            print(f"Better than reference: {better} ({better / len(gaps) * 100:.1f}%)")
            print(f"Equal to reference: {equal} ({equal / len(gaps) * 100:.1f}%)")
            print(f"Worse than reference: {worse} ({worse / len(gaps) * 100:.1f}%)")

            # Simple histogram
            print("\nGap Histogram:")
            # Create bins
            bins = [-0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.05, 0.10, 0.20, 0.50, 1.0]
            counts = [0] * (len(bins) + 1)
            for g in gaps:
                placed = False
                for i, b in enumerate(bins):
                    if g <= b:
                        counts[i] += 1
                        placed = True
                        break
                if not placed:
                    counts[-1] += 1

            labels = [f"<= {b:.2f}" for b in bins] + [f"> {bins[-1]:.2f}"]
            for label, count in zip(labels, counts):
                if count > 0:
                    bar = "#" * count
                    print(f"{label:>10}: {count:3d} {bar}")

        # Save detailed results to CSV
        if self.save_dir:
            os.makedirs(self.save_dir, exist_ok=True)
            csv_path = os.path.join(self.save_dir, f"benchmark_results_step_{step}.csv")
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "instance_name",
                        "instance_hash",
                        "policy_makespan",
                        "reference_makespan",
                        "gap",
                    ],
                )
                writer.writeheader()
                writer.writerows(detailed_results)
            print(f"Detailed results saved to {csv_path}")

        if self.save_best_model:
            current_metric = metrics.get(self.metric_key)
            if current_metric is None:
                raise KeyError(
                    f"Metric '{self.metric_key}' not found in benchmark metrics."
                )
            is_better = (
                current_metric < self.best_metric
                if self.metric_mode == "min"
                else current_metric > self.best_metric
            )
            if is_better:
                self.best_metric = current_metric
                self.best_step = step
                self._save_model(policy_module, step, current_metric)
                self.logger.log_custom_metrics(
                    {f"benchmark/best_{self.metric_key}": self.best_metric},
                    step=step,
                )
                print(
                    f"New best benchmark {self.metric_key}: "
                    f"{self.best_metric:.4f} at step {step}"
                )

        return metrics

    def _save_model(self, model, step: int, metric_value: float):
        """Save the model checkpoint for benchmark tracking."""
        import os

        assert self.save_dir is not None
        os.makedirs(self.save_dir, exist_ok=True)

        if isinstance(model, tuple):
            separate_models = model[0]
            main_model = model[1]

            for key, small_model in separate_models.items():
                checkpoint_path = os.path.join(self.save_dir, f"{key}_best_model.pt")
                torch.save(small_model.state_dict(), checkpoint_path)

            main_checkpoint_path = os.path.join(self.save_dir, "best_model.pt")
            torch.save(main_model.state_dict(), main_checkpoint_path)
        else:
            checkpoint_path = os.path.join(self.save_dir, "best_model.pt")
            torch.save(model.state_dict(), checkpoint_path)

        metadata_path = os.path.join(self.save_dir, "best_model_metadata.txt")
        with open(metadata_path, "w", encoding="utf-8") as f:
            f.write(f"Best {self.metric_key}: {metric_value:.6f}\n")
            f.write(f"Step: {step}\n")
            f.write(f"Mode: {self.metric_mode}\n")

    def get_best_metric(self) -> float:
        """Return the best tracked metric value."""
        return self.best_metric

    def get_best_step(self) -> int | None:
        """Return the training step of the best metric."""
        return self.best_step


def create_evaluator(
    env: TransformedEnv,
    logger: TrainingLogger,
    cfg: DictConfig,
    dispatcher: DispatcherBase = None,
) -> BaseEvaluator:
    """
    Factory function to create evaluator based on configuration.

    Args:
        env: The environment to evaluate on
        logger: Logger for recording evaluation metrics
        cfg: Configuration containing evaluation settings
            evaluation:
              type: "standard", "benchmark",, "multi_agent", "multi_agent_advantage", or "none"
              max_steps: int (default: 1000)
              lower_bound: float (optional, for standard evaluator)
              save_best_model: bool (default: False)
              save_dir: str (optional, required if save_best_model=True)
              metric_mode: "min" or "max" (default: "min")
              metric_key: str (default: "makespan")
              # For benchmark evaluator:
              use_generator: bool (default: False) - whether to use instance generator
              use_heuristic_reference: bool (default: False) - whether to use on-the-fly heuristic as reference
              generator_type: str (default: "random_uniform") - type of generator if use_generator=True
              generator_kwargs: dict (optional) - kwargs for generator
              reference_makespan: float (optional) - fixed reference makespan for generator-based evaluation
              heuristic_name: str (default: "mwr") - heuristic to use for on-the-fly reference computation
              dataset_filename: str (default: "jssp_instances/heuristic_solutions/solved_instances.json") - for dataset-based evaluation
              n_instances: int (default: 100)
              benchmark_heuristic: str (default: "mwkr") - for dataset-based evaluation with pre-computed solutions

    Returns:
        Configured evaluator instance
    """
    eval_type = cfg.evaluation.get("type", "standard")
    print(f"===== Using Evaluator: {eval_type} =====")
    if eval_type == "standard":
        return StandardEvaluator(
            env=env,
            logger=logger,
            max_steps=cfg.evaluation.get("max_steps", 1000),
            lower_bound=cfg.evaluation.get("lower_bound", None),
            save_best_model=cfg.evaluation.get("save_best_model", False),
            save_dir=cfg.get("save_dir", None),
            metric_mode=cfg.evaluation.get("metric_mode", "min"),
            metric_key=cfg.evaluation.get("metric_key", "makespan"),
        )
    elif eval_type == "benchmark":
        # Determine instance generator based on configuration

        # Check if using generator
        if cfg.evaluation.get("use_generator", False):
            from jssp_core.instances.generators import initialize_instance_generator

            # Get environment dimensions
            base_env = _unwrap_base_env(env)
            num_jobs = getattr(base_env, "num_jobs", None)
            num_machines = getattr(base_env, "num_machines", None)

            if num_jobs is None or num_machines is None:
                raise ValueError(
                    "Cannot infer instance size from environment for generator. "
                    "Please configure num_jobs and num_machines."
                )

            generator_type = cfg.evaluation.get("generator_type", "random_uniform")
            generator_kwargs = cfg.evaluation.get("generator_kwargs", {})

            generator = initialize_instance_generator(
                generator_type,
                num_jobs=num_jobs,
                num_machines=num_machines,
                generator_kwargs=generator_kwargs,
            )

            # Check if using on-the-fly heuristic reference
            if cfg.evaluation.get("use_heuristic_reference", False):
                heuristic_name = cfg.evaluation.get("heuristic_name", None)
                instance_gen = heuristic_instance_generator(
                    generator=generator,
                    heuristic_name=heuristic_name,
                )
            else:
                instance_gen = simple_instance_generator(
                    generator=generator,
                    reference_makespan=cfg.evaluation.get("reference_makespan", None),
                )
        else:
            # Use dataset file (default behavior)
            instance_size = cfg.evaluation.get("instance_size", None)
            if instance_size is not None:
                instance_size = tuple(instance_size)
            else:
                # Infer from environment
                base_env = _unwrap_base_env(env)
                num_jobs = getattr(base_env, "num_jobs", None)
                num_machines = getattr(base_env, "num_machines", None)
                if num_jobs is not None and num_machines is not None:
                    instance_size = (num_jobs, num_machines)
                else:
                    raise ValueError(
                        "Could not infer instance_size. Please specify evaluation.instance_size"
                    )

            dataset_filename = cfg.evaluation.get("dataset_filename", None)
            benchmark_heuristic = cfg.evaluation.get("benchmark_heuristic", None)

        return BenchmarkEvaluator(
            env=env,
            logger=logger,
            instance_generator=instance_gen,
            n_instances=cfg.evaluation.get("n_instances", 100),
            max_steps=cfg.evaluation.get("max_steps", 1000),
            save_best_model=cfg.evaluation.get("save_best_model", False),
            save_dir=cfg.get("save_dir", None),
            metric_mode=cfg.evaluation.get("metric_mode", "min"),
            metric_key=cfg.evaluation.get("metric_key", "mean_gap"),
        )
    elif eval_type == "none":
        return NoOpEvaluator(env=env, logger=logger)
    else:
        raise ValueError(f"Unknown evaluation type: {eval_type}")
