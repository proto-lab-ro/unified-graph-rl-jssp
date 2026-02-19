import argparse
import datetime
import os
import random
import time
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any
from uuid import uuid4

from omegaconf import DictConfig, OmegaConf

from jssp_core.benchmark.duckdb_result_sink import DuckDBResultSink, ResultSink
from jssp_core.benchmark.records import (
    BenchmarkRecord,
    BenchmarkRunRecord,
    SolverConfigRecord,
    SolverType,
)
from jssp_core.benchmark.sqlserver_result_sink import SqlServerResultSink
from jssp_core.instances.generators import initialize_instance_generator
from jssp_core.reproducibility import set_seed
from jssp_core.solver.heuristics import job_heuristic_factory

# from jssp_core.instances.generators import FileInstanceGenerator
from jssp_core.solver.optimal import JSSPOptimalSolver
from jssp_gnn.solver import GnnMatrixSolver


START_TIME = datetime.datetime.now(datetime.UTC)

RUN_ID = START_TIME.strftime("run_%Y%m%d_%H%M%S")
DB_PATH = Path("local_benchmarks.duckdb")
MODEL_STORE_DIR = Path("model_store")
SEED = 42


class LazySolver:
    def __init__(self, name: str, factory: Callable[[], Any], solver_type: SolverType):
        self.name = name
        self.factory = factory
        self._solver_type = solver_type

    def get_name(self) -> str:
        return self.name

    def get_type(self) -> SolverType:
        return self._solver_type

    def load(self) -> Any:
        return self.factory()


def resolve_model_path(model_name: str) -> Path:
    p = Path(model_name)
    if p.exists():
        return p

    # Check in model store
    p_store = MODEL_STORE_DIR / model_name
    if p_store.exists():
        return p_store

    # Check in model store with .tar.gz
    p_store_tgz = MODEL_STORE_DIR / f"{model_name}.tar.gz"
    if p_store_tgz.exists():
        return p_store_tgz

    return p


def build_solvers(
    explicit_models: list[str] | None = None, random_nth_step: int | None = None
) -> list[LazySolver]:
    if explicit_models:
        tar_gz_files = [resolve_model_path(p) for p in explicit_models]
    else:
        tar_gz_files = sorted(Path(MODEL_STORE_DIR).glob("*.tar.gz"))

    print("Found archives:", tar_gz_files)

    heuristic_solvers = []
    for h in [
        "spt",
        "smpt",
        "mwr",
        "fddmwr",
        "mor",
        "random",
    ]:

        def factory(h=h):
            return job_heuristic_factory(h, random_nth_step)

        name = h
        if random_nth_step is not None:
            name += f"__rnd{random_nth_step}"
        heuristic_solvers.append(LazySolver(name, factory, SolverType.HEURISTIC))

    gnn_solvers = []
    for tar_path in tar_gz_files:
        # Use filename stem as name
        base_name = tar_path.name.replace(".tar.gz", "").replace(".tgz", "")
        checkpoint_filename = "policy_module_final.pt"
        checkpoint_suffix = checkpoint_filename.replace(".pt", "")
        name = f"{base_name}__{checkpoint_suffix}"

        if random_nth_step is not None:
            name += f"__rnd{random_nth_step}"

        factory_gnn = partial(
            GnnMatrixSolver.from_package,
            str(tar_path),
            device="cpu",
            max_steps=2000,
            force_extract=True,
            checkpoint_filename=checkpoint_filename,
            random_nth_step=random_nth_step,
        )
        gnn_solvers.append(LazySolver(name, factory_gnn, SolverType.GNN))

    def optimal_factory():
        return JSSPOptimalSolver(time_limit_seconds=2)

    optimal_solver = LazySolver("optimal", optimal_factory, SolverType.OPTIMAL)

    if explicit_models:
        return gnn_solvers

    solvers = [*gnn_solvers, *heuristic_solvers, optimal_solver]
    # solvers = heuristic_solvers
    return solvers


def write_matrix_solver_conf(sink: ResultSink, solver: GnnMatrixSolver):
    print(f"Using GNN solver with model: {solver.name}")

    cfg_pruned = solver.cfg.copy()

    for k in ("log_dir", "save_dir"):
        if k in cfg_pruned:
            del cfg_pruned[k]

    config_params = OmegaConf.to_container(
        cfg_pruned,
        resolve=True,
        throw_on_missing=True,
    )

    sink.write_solver_config(
        SolverConfigRecord(
            solver_id=solver.name,
            solver_type=SolverType.GNN,
            config_params=config_params,
        )
    )


def build_sink(use_sql_server: bool) -> ResultSink:
    if use_sql_server:
        sql_conn_str = os.environ.get("SQL_SERVER_CONN_STR", "")
        if not sql_conn_str:
            raise ValueError(
                "SQL_SERVER_CONN_STR must be set when USE_SQL_SERVER is true"
            )
        return SqlServerResultSink(sql_conn_str)

    return DuckDBResultSink(DB_PATH)


def main():
    parser = argparse.ArgumentParser(description="Run JSSP Benchmark")
    parser.add_argument(
        "--models", nargs="*", help="List of model paths to evaluate (tar.gz files)"
    )
    parser.add_argument(
        "--random-nth-step",
        type=int,
        default=None,
        help="Pick random action every nth step",
    )
    args = parser.parse_args()

    set_seed(SEED)

    # Define multiple generator configurations
    generator_configs = [
        {
            "generator": "pickle",
            "num_jobs": 0,
            "num_machines": 0,
            "generator_kwargs": {"file_path": "benchmark_results/bm_20x15-100x20.pkl"},
        },
        {
            "generator": "pickle",
            "num_jobs": 0,
            "num_machines": 0,
            "generator_kwargs": {"file_path": "benchmark_results/sm_5x5-30x10.pkl"},
        },
        {
            # Custom Random 100 each 6x6, 10x10, 15x15, 20x20, 30x20, d(1-99) Pickle Instances
            "generator": "pickle",
            "num_jobs": 0,
            "num_machines": 0,
            "generator_kwargs": {
                "file_path": "benchmark_results/251211_benchmark_instances_v1.pkl"
            },
        },
        {
            # LA Bechmark Instances
            "generator": "from_file",
            "num_jobs": 0,
            "num_machines": 0,
            "generator_kwargs": {"file_pattern": "jssp_instances/la*"},
        },
        {
            # TA Benchmark Instances
            "generator": "from_file",
            "num_jobs": 0,
            "num_machines": 0,
            "generator_kwargs": {"file_pattern": "jssp_instances/ta*"},
        },
    ]

    # Initialize all generators and collect instances
    all_instances = []
    for config in generator_configs:
        print(f"Initializing generator: {config}")
        try:
            gen = initialize_instance_generator(**config)
            instances = list(gen)
            print(f"  Loaded {len(instances)} instances")
            all_instances.extend(instances)
        except Exception as e:
            print(f"  Failed to load generator {config}: {e}")

    print(f"Total instances loaded: {len(all_instances)}")

    solvers = build_solvers(args.models, args.random_nth_step)

    use_sql_server = os.environ.get("USE_SQL_SERVER", "false").lower() == "true"
    sink_instance = build_sink(use_sql_server)
    print(f"Using sink: {type(sink_instance).__name__}")

    with sink_instance as sink:
        # Load completed runs using instance_hash instead of instance_id
        # Querying for ALL completed runs to skip duplicates
        completed = sink.get_completed_runs()
        print(f"Found {len(completed)} completed runs in DB")

        worker_id = str(uuid4())
        print(f"Worker ID: {worker_id}")

        # Shuffle solvers to distribute work
        random.shuffle(solvers)

        # Execute tasks
        benchmark_run_record = BenchmarkRunRecord(
            benchmark_run_id=RUN_ID,
            timestamp=START_TIME,
            seed=SEED,
            generator_name="multiple_generators",
            generator_params={"configs": generator_configs},
        )
        sink.write_benchmark_run(benchmark_run_record)

        for lazy_solver in solvers:
            solver_name = lazy_solver.get_name()

            # Identify pending instances for this solver
            pending_instances = []
            for instance in all_instances:
                if (instance.get_hash(), solver_name) not in completed:
                    pending_instances.append(instance)

            if not pending_instances:
                print(f"Solver {solver_name} has no pending instances. Skipping.")
                continue

            print(f"Attempting to lock solver {solver_name}...")
            try:
                locked = sink.try_lock_solver(solver_name, worker_id)
            except Exception as e:
                print(f"Failed to check lock for solver {solver_name}: {e}")
                locked = False

            if locked:
                print(f"Locked solver {solver_name}. Loading solver...")
                try:
                    # Load the actual solver
                    solver = lazy_solver.load()

                    # Write config if applicable
                    if isinstance(solver, GnnMatrixSolver) and isinstance(
                        solver.cfg, DictConfig
                    ):
                        write_matrix_solver_conf(sink, solver)

                    print(
                        f"Solver loaded. Processing {len(pending_instances)} instances."
                    )

                    buffer = []
                    BATCH_SIZE = 50

                    for i, instance in enumerate(pending_instances):
                        instance_hash = instance.get_hash()
                        instance_id = getattr(
                            instance, "name", f"unknown_{instance_hash[:8]}"
                        )

                        print(
                            f"[{i + 1}/{len(pending_instances)}] Running {solver_name} on {instance_id} ({instance.num_jobs()}x{instance.num_machines()})"
                        )

                        run_id = uuid4().hex
                        computation_time = 0.0
                        try:
                            start_time = time.perf_counter()
                            output = solver.solve_with_info(instance)
                            end_time = time.perf_counter()
                            computation_time = end_time - start_time

                            record = BenchmarkRecord(
                                run_id=run_id,
                                benchmark_run_id=benchmark_run_record.benchmark_run_id,
                                instance_id=instance_id,
                                instance_hash=instance_hash,
                                solver_id=solver_name,
                                solver_type=solver.get_type(),
                                instance_size=(
                                    instance.num_jobs(),
                                    instance.num_machines(),
                                ),
                                makespan=output.solution.get_makespan(),
                                computation_time_seconds=computation_time,
                                solution=output.solution.get_action_sequence(),
                                additional_metrics=output.info,
                                error=None,
                            )

                        except Exception as e:
                            print(f"  Failed: {e}")
                            # Optionally write error record
                            record = BenchmarkRecord(
                                run_id=run_id,
                                benchmark_run_id=benchmark_run_record.benchmark_run_id,
                                instance_id=instance_id,
                                instance_hash=instance_hash,
                                solver_id=solver_name,
                                solver_type=solver.get_type(),
                                instance_size=(
                                    instance.num_jobs(),
                                    instance.num_machines(),
                                ),
                                makespan=0,
                                computation_time_seconds=0.0,
                                solution=None,
                                additional_metrics={},
                                error=str(e),
                            )

                        buffer.append(record)
                        if len(buffer) >= BATCH_SIZE:
                            print(f"Writing batch of {len(buffer)} records to sink...")
                            try:
                                sink.write_records(buffer)
                            except Exception as e:
                                print(f"Failed to write batch to sink: {e}")
                            buffer.clear()

                    if buffer:
                        try:
                            sink.write_records(buffer)
                        except Exception as e:
                            print(f"Failed to write final batch to sink: {e}")
                        buffer.clear()
                except Exception as e:
                    print(f"Error running solver {solver_name}: {e}")
                finally:
                    try:
                        sink.unlock_solver(solver_name, worker_id)
                        print(f"Unlocked solver {solver_name}.")
                    except Exception as e:
                        print(f"Failed to unlock solver {solver_name}: {e}")
            else:
                print(f"Solver {solver_name} is locked by another worker. Skipping.")


if __name__ == "__main__":
    main()
