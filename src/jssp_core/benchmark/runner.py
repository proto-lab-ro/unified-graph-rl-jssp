import time
from uuid import uuid4

from jssp_core.benchmark.duckdb_result_sink import ResultSink
from jssp_core.benchmark.records import (
    BenchmarkRecord,
    BenchmarkRunRecord,
    RewardRecord,
)
from jssp_core.instances.generators import InstanceGenerator
from jssp_core.solver.base import SolverProtocol


class BenchmarkRunner:
    """
    Unified benchmark runner.
    Runs a set of solvers on a set of instances provided by an InstanceGenerator.
    """

    def __init__(self, sink: ResultSink, verbose: bool = True):
        self.sink = sink
        self.verbose = verbose

    def run(
        self,
        solvers: list[SolverProtocol],
        instance_generator: list[InstanceGenerator],
        benchmark_run_record: BenchmarkRunRecord,
    ):
        """
        Run the benchmark.
        """
        self.sink.write_benchmark_run(benchmark_run_record)

        generators = (
            instance_generator
            if isinstance(instance_generator, list)
            else [instance_generator]
        )

        instance_count = 0
        for generator in generators:
            for instance in generator:
                instance_id = getattr(instance, "name", f"instance_{instance_count}")
                instance_count += 1

                if self.verbose:
                    print(
                        f"Benchmarking instance: {instance_id} ({instance.num_jobs()}x{instance.num_machines()})"
                    )

                for solver in solvers:
                    run_id = uuid4().hex
                    result = None
                    computation_time = 0.0
                    try:
                        if self.verbose:
                            print(f"  Running {solver.name}...", end="", flush=True)

                        start_time = time.perf_counter()
                        output = solver.solve_with_info(instance)
                        end_time = time.perf_counter()
                        computation_time = end_time - start_time

                        self.sink.write_record(
                            BenchmarkRecord(
                                run_id=run_id,
                                benchmark_run_id=benchmark_run_record.benchmark_run_id,
                                instance_id=instance_id,
                                instance_hash=instance.get_hash(),
                                solver_id=solver.name,
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
                        )
                        result = output  # For reward logging below

                    except Exception as e:
                        print(f"  Failed: {e}")
                        self.sink.write_record(
                            BenchmarkRecord(
                                run_id=run_id,
                                benchmark_run_id=benchmark_run_record.benchmark_run_id,
                                instance_id=instance_id,
                                instance_hash=instance.get_hash(),
                                solver_id=solver.name,
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
                        )

                    if result and result.info.get("total_return") is not None:
                        self.sink.write_reward(
                            RewardRecord(
                                run_id=run_id,
                                reward=result.info["total_return"],
                            )
                        )

                    if self.verbose and result:
                        print(
                            f" Done. Makespan: {result.solution.get_makespan()} ({computation_time:.4f}s)"
                        )
