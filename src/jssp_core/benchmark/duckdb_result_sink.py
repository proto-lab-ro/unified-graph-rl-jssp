import json
from pathlib import Path
from typing import Protocol

import duckdb

from jssp_core.benchmark.records import (
    BenchmarkRecord,
    BenchmarkRunRecord,
    RewardRecord,
    SolverConfigRecord,
)
from jssp_core.benchmark.sql_schema import (
    BENCHMARK_RECORDS_SCHEMA,
    BENCHMARK_RUN_SCHEMA,
    REWARDS_SCHEMA,
    SOLVER_CONFIG_SCHEMA,
    SOLVER_LOCKS_SCHEMA,
)


class ResultSink(Protocol):
    def write_record(self, result: BenchmarkRecord) -> None: ...
    def write_records(self, results: list[BenchmarkRecord]) -> None: ...
    def write_reward(self, reward_record: RewardRecord) -> None: ...
    def write_benchmark_run(self, benchmark_run_record: BenchmarkRunRecord) -> None: ...
    def write_solver_config(self, solver_config: SolverConfigRecord) -> None: ...
    def close(self) -> None: ...


class DuckDBResultSink:
    def __init__(self, db_path: str | Path):
        self.conn = duckdb.connect(str(db_path))
        self._init_schema()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        # returning False propagates exceptions
        return False

    def close(self) -> None:
        if self.conn is not None:
            self.conn.close()
            self.conn = None

    def _init_schema(self) -> None:
        for stmt in (
            BENCHMARK_RECORDS_SCHEMA,
            REWARDS_SCHEMA,
            BENCHMARK_RUN_SCHEMA,
            SOLVER_CONFIG_SCHEMA,
            SOLVER_LOCKS_SCHEMA,
        ):
            self.conn.execute(stmt)

    def get_completed_runs(self) -> set[tuple[str, str]]:
        """
        Get a set of (instance_hash, solver_id) tuples for completed runs.
        """
        try:
            rows = self.conn.execute(
                """
                SELECT instance_hash, solver_id
                FROM benchmark_records
                WHERE error IS NULL
                """
            ).fetchall()
            return {(row[0], row[1]) for row in rows}
        except duckdb.CatalogException:
            # Table might not exist yet
            return set()

    def is_run_completed(self, instance_hash: str, solver_id: str) -> bool:
        """
        Check if a specific run is completed.
        """
        try:
            row = self.conn.execute(
                """
                SELECT 1
                FROM benchmark_records
                WHERE instance_hash = ? AND solver_id = ? AND error IS NULL
                LIMIT 1
                """,
                (instance_hash, solver_id),
            ).fetchone()
            return row is not None
        except duckdb.CatalogException:
            return False

    def try_lock_solver(self, solver_id: str, worker_id: str) -> bool:
        """
        Try to acquire a lock for a solver.
        Returns True if successful, False if already locked.
        """
        try:
            self.conn.execute(
                "INSERT INTO solver_locks (solver_id, worker_id, lock_time) VALUES (?, ?, current_timestamp)",
                (solver_id, worker_id),
            )
            return True
        except duckdb.ConstraintException:
            return False
        except duckdb.CatalogException:
            # Table might not exist yet
            return False

    def unlock_solver(self, solver_id: str, worker_id: str) -> None:
        """
        Release the lock for a solver.
        """
        try:
            self.conn.execute(
                "DELETE FROM solver_locks WHERE solver_id = ? AND worker_id = ?",
                (solver_id, worker_id),
            )
        except duckdb.CatalogException:
            pass

    def write_record(self, record: BenchmarkRecord) -> None:
        self.conn.execute(
            """
            INSERT INTO benchmark_records (
                run_id,
                benchmark_run_id,
                instance_id,
                instance_hash,
                solver_id,
                solver_type,
                num_jobs,
                num_machines,
                makespan,
                computation_time_seconds,
                solution,
                additional_metrics,
                error
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.run_id,
                record.benchmark_run_id,
                record.instance_id,
                record.instance_hash,
                record.solver_id,
                record.solver_type.value
                if hasattr(record.solver_type, "value")
                else str(record.solver_type),
                record.instance_size[0],
                record.instance_size[1],
                record.makespan,
                record.computation_time_seconds,
                json.dumps(record.solution) if record.solution is not None else None,
                json.dumps(record.additional_metrics),
                record.error,
            ),
        )

    def write_records(self, records: list[BenchmarkRecord]) -> None:
        data = []
        for record in records:
            data.append(
                (
                    record.run_id,
                    record.benchmark_run_id,
                    record.instance_id,
                    record.instance_hash,
                    record.solver_id,
                    record.solver_type.value
                    if hasattr(record.solver_type, "value")
                    else str(record.solver_type),
                    record.instance_size[0],
                    record.instance_size[1],
                    record.makespan,
                    record.computation_time_seconds,
                    json.dumps(record.solution)
                    if record.solution is not None
                    else None,
                    json.dumps(record.additional_metrics),
                    record.error,
                )
            )

        self.conn.executemany(
            """
            INSERT INTO benchmark_records (
                run_id,
                benchmark_run_id,
                instance_id,
                instance_hash,
                solver_id,
                solver_type,
                num_jobs,
                num_machines,
                makespan,
                computation_time_seconds,
                solution,
                additional_metrics,
                error
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            data,
        )

    def write_reward(self, reward_record: RewardRecord) -> None:
        """Write a reward record to the database."""
        self.conn.execute(
            """
            INSERT INTO rewards (
                run_id,
                reward
            ) VALUES (?, ?)
            """,
            (
                reward_record.run_id,
                reward_record.reward,
            ),
        )

    def write_benchmark_run(self, benchmark_run_record: BenchmarkRunRecord) -> None:
        existing = self.conn.execute(
            "SELECT 1 FROM benchmark_run WHERE benchmark_run_id = ?",
            [benchmark_run_record.benchmark_run_id],
        ).fetchone()

        if existing:
            return

        self.conn.execute(
            """
            INSERT INTO benchmark_run (
                benchmark_run_id,
                timestamp,
                seed,
                generator_name,
                generator_params
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (
                benchmark_run_record.benchmark_run_id,
                benchmark_run_record.timestamp,
                benchmark_run_record.seed,
                benchmark_run_record.generator_name,
                json.dumps(benchmark_run_record.generator_params),
            ),
        )

    def write_solver_config(self, solver_config: SolverConfigRecord) -> None:
        existing = self.conn.execute(
            "SELECT 1 FROM solver_config WHERE solver_id = ?",
            [solver_config.solver_id],
        ).fetchone()

        if existing:
            return

        self.conn.execute(
            """
            INSERT INTO solver_config (
                solver_id,
                solver_type,
                config_params
            ) VALUES (?, ?, ?)
            """,
            (
                solver_config.solver_id,
                solver_config.solver_type.value
                if hasattr(solver_config.solver_type, "value")
                else str(solver_config.solver_type),
                json.dumps(solver_config.config_params),
            ),
        )
