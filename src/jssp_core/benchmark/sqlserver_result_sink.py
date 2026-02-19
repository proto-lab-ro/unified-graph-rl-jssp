import functools
import json
import time

import pyodbc

from jssp_core.benchmark.records import (
    BenchmarkRecord,
    BenchmarkRunRecord,
    RewardRecord,
    SolverConfigRecord,
)


# Define SQL Server schemas locally to avoid polluting the common schema file
# with database-specific syntax.

BENCHMARK_RECORDS_SCHEMA_SQLSERVER = """
IF OBJECT_ID('dbo.benchmark_records', 'U') IS NULL
CREATE TABLE dbo.benchmark_records (
    run_id NVARCHAR(MAX) NOT NULL,
    benchmark_run_id NVARCHAR(450) NOT NULL,
    instance_id NVARCHAR(MAX) NOT NULL,
    instance_hash NVARCHAR(450) NOT NULL,
    solver_id NVARCHAR(450) NOT NULL,
    solver_type NVARCHAR(MAX) NOT NULL,

    num_jobs INT NOT NULL,
    num_machines INT NOT NULL,

    makespan INT,
    computation_time_seconds FLOAT,

    solution NVARCHAR(MAX),
    additional_metrics NVARCHAR(MAX),
    error NVARCHAR(MAX),

    CONSTRAINT PK_benchmark_records PRIMARY KEY (benchmark_run_id, instance_hash, solver_id)
);
"""

REWARDS_SCHEMA_SQLSERVER = """
IF OBJECT_ID('dbo.rewards', 'U') IS NULL
CREATE TABLE dbo.rewards (
    run_id NVARCHAR(450) NOT NULL,
    reward FLOAT NOT NULL
    -- No primary key defined in original schema, but run_id might be good if unique per reward?
    -- The original schema didn't have one.
);
"""

BENCHMARK_RUN_SCHEMA_SQLSERVER = """
IF OBJECT_ID('dbo.benchmark_run', 'U') IS NULL
CREATE TABLE dbo.benchmark_run (
    benchmark_run_id NVARCHAR(450) NOT NULL,
    timestamp NVARCHAR(MAX) NOT NULL,
    seed INT NOT NULL,
    generator_name NVARCHAR(MAX) NOT NULL,
    generator_params NVARCHAR(MAX) NOT NULL,

    CONSTRAINT PK_benchmark_run PRIMARY KEY (benchmark_run_id)
);
"""

SOLVER_CONFIG_SCHEMA_SQLSERVER = """
IF OBJECT_ID('dbo.solver_config', 'U') IS NULL
CREATE TABLE dbo.solver_config (
    solver_id NVARCHAR(450) NOT NULL,
    solver_type NVARCHAR(MAX) NOT NULL,
    config_params NVARCHAR(MAX) NOT NULL,

    CONSTRAINT PK_solver_config PRIMARY KEY (solver_id)
);
"""

SOLVER_LOCKS_SCHEMA_SQLSERVER = """
IF OBJECT_ID('dbo.solver_locks', 'U') IS NULL
CREATE TABLE dbo.solver_locks (
    solver_id NVARCHAR(450) NOT NULL,
    worker_id NVARCHAR(MAX) NOT NULL,
    lock_time DATETIME NOT NULL,
    PRIMARY KEY (solver_id)
);
"""


def retry_on_db_error(func):
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        max_retries = 3
        last_error = None

        for attempt in range(max_retries):
            try:
                if self.conn is None:
                    self._reconnect()
                return func(self, *args, **kwargs)
            except pyodbc.Error as e:
                last_error = e
                if attempt < max_retries - 1:
                    try:
                        time.sleep(15)
                        self._reconnect()
                    except Exception:
                        pass

        if last_error:
            raise last_error

    return wrapper


class SqlServerResultSink:
    def __init__(self, connection_string: str):
        """
        Initialize the SQL Server result sink.

        Args:
            connection_string: ODBC connection string for SQL Server.
                             e.g. 'DRIVER={ODBC Driver 17 for SQL Server};SERVER=localhost;DATABASE=mydb;UID=user;PWD=password'
        """
        self.connection_string = connection_string
        self.conn = pyodbc.connect(self.connection_string)
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

    def _reconnect(self) -> None:
        """Attempt to reconnect to the database."""
        if self.conn is not None:
            try:
                self.conn.close()
            except Exception:
                pass
        self.conn = pyodbc.connect(self.connection_string)

    def _init_schema(self) -> None:
        cursor = self.conn.cursor()
        try:
            for stmt in (
                BENCHMARK_RECORDS_SCHEMA_SQLSERVER,
                REWARDS_SCHEMA_SQLSERVER,
                BENCHMARK_RUN_SCHEMA_SQLSERVER,
                SOLVER_CONFIG_SCHEMA_SQLSERVER,
                SOLVER_LOCKS_SCHEMA_SQLSERVER,
            ):
                cursor.execute(stmt)
            self.conn.commit()
        finally:
            cursor.close()

    def get_completed_runs(self) -> set[tuple[str, str]]:
        """
        Get a set of (instance_hash, solver_id) tuples for completed runs.
        """
        cursor = self.conn.cursor()
        try:
            # Check if table exists first to avoid error
            if (
                cursor.execute(
                    "SELECT OBJECT_ID('dbo.benchmark_records', 'U')"
                ).fetchval()
                is None
            ):
                return set()

            rows = cursor.execute(
                """
                SELECT instance_hash, solver_id
                FROM benchmark_records
                WHERE error IS NULL
                """
            ).fetchall()
            return {(row[0], row[1]) for row in rows}
        finally:
            cursor.close()

    def is_run_completed(self, instance_hash: str, solver_id: str) -> bool:
        """
        Check if a specific run is completed.
        """
        cursor = self.conn.cursor()
        try:
            # Check if table exists first
            if (
                cursor.execute(
                    "SELECT OBJECT_ID('dbo.benchmark_records', 'U')"
                ).fetchval()
                is None
            ):
                return False

            row = cursor.execute(
                """
                SELECT 1
                FROM benchmark_records
                WHERE instance_hash = ? AND solver_id = ? AND error IS NULL
                """,
                (instance_hash, solver_id),
            ).fetchone()
            return row is not None
        finally:
            cursor.close()

    def try_lock_solver(self, solver_id: str, worker_id: str) -> bool:
        """
        Try to acquire a lock for a solver.
        Returns True if successful, False if already locked.
        """
        cursor = self.conn.cursor()
        try:
            cursor.execute(
                "INSERT INTO solver_locks (solver_id, worker_id, lock_time) VALUES (?, ?, GETUTCDATE())",
                (solver_id, worker_id),
            )
            self.conn.commit()
            return True
        except pyodbc.IntegrityError:
            return False
        finally:
            cursor.close()

    def unlock_solver(self, solver_id: str, worker_id: str) -> None:
        """
        Release the lock for a solver.
        """
        cursor = self.conn.cursor()
        try:
            cursor.execute(
                "DELETE FROM solver_locks WHERE solver_id = ? AND worker_id = ?",
                (solver_id, worker_id),
            )
            self.conn.commit()
        finally:
            cursor.close()

    @retry_on_db_error
    def write_record(self, record: BenchmarkRecord) -> None:
        cursor = self.conn.cursor()
        try:
            cursor.execute(
                """
                IF NOT EXISTS (SELECT 1 FROM benchmark_records WHERE benchmark_run_id = ? AND instance_hash = ? AND solver_id = ?)
                BEGIN
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
                END
                """,
                (
                    record.benchmark_run_id,
                    record.instance_hash,
                    record.solver_id,
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
                ),
            )
            self.conn.commit()
        finally:
            cursor.close()

    @retry_on_db_error
    def write_records(self, records: list[BenchmarkRecord]) -> None:
        cursor = self.conn.cursor()
        try:
            for record in records:
                cursor.execute(
                    """
                    IF NOT EXISTS (SELECT 1 FROM benchmark_records WHERE benchmark_run_id = ? AND instance_hash = ? AND solver_id = ?)
                    BEGIN
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
                    END
                    """,
                    (
                        record.benchmark_run_id,
                        record.instance_hash,
                        record.solver_id,
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
                    ),
                )
            self.conn.commit()
        finally:
            cursor.close()

    @retry_on_db_error
    def write_reward(self, reward_record: RewardRecord) -> None:
        """Write a reward record to the database."""
        cursor = self.conn.cursor()
        try:
            cursor.execute(
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
            self.conn.commit()
        finally:
            cursor.close()

    @retry_on_db_error
    def write_benchmark_run(self, benchmark_run_record: BenchmarkRunRecord) -> None:
        cursor = self.conn.cursor()
        try:
            cursor.execute(
                """
                IF NOT EXISTS (SELECT 1 FROM benchmark_run WHERE benchmark_run_id = ?)
                BEGIN
                    INSERT INTO benchmark_run (
                        benchmark_run_id,
                        timestamp,
                        seed,
                        generator_name,
                        generator_params
                    ) VALUES (?, ?, ?, ?, ?)
                END
                """,
                (
                    benchmark_run_record.benchmark_run_id,
                    benchmark_run_record.benchmark_run_id,
                    benchmark_run_record.timestamp,
                    benchmark_run_record.seed,
                    benchmark_run_record.generator_name,
                    json.dumps(benchmark_run_record.generator_params),
                ),
            )
            self.conn.commit()
        finally:
            cursor.close()

    @retry_on_db_error
    def write_solver_config(self, solver_config: SolverConfigRecord) -> None:
        """Write a solver configuration record to the database."""
        cursor = self.conn.cursor()
        try:
            cursor.execute(
                """
                IF NOT EXISTS (SELECT 1 FROM solver_config WHERE solver_id = ?)
                BEGIN
                    INSERT INTO solver_config (
                        solver_id,
                        solver_type,
                        config_params
                    ) VALUES (?, ?, ?)
                END
                """,
                (
                    solver_config.solver_id,
                    solver_config.solver_id,
                    solver_config.solver_type.value
                    if hasattr(solver_config.solver_type, "value")
                    else str(solver_config.solver_type),
                    json.dumps(solver_config.config_params),
                ),
            )
            self.conn.commit()
        finally:
            cursor.close()
