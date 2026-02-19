BENCHMARK_RECORDS_SCHEMA = """
            CREATE TABLE IF NOT EXISTS benchmark_records (
                run_id TEXT NOT NULL,
                benchmark_run_id TEXT NOT NULL,
                instance_id TEXT NOT NULL,
                instance_hash TEXT NOT NULL,
                solver_id TEXT NOT NULL,
                solver_type TEXT NOT NULL,

                num_jobs INTEGER NOT NULL,
                num_machines INTEGER NOT NULL,

                makespan INTEGER,
                computation_time_seconds DOUBLE,

                solution JSON,
                additional_metrics JSON,
                error TEXT,

                PRIMARY KEY (benchmark_run_id, instance_hash, solver_id)
            );
            """

REWARDS_SCHEMA = """
            CREATE TABLE IF NOT EXISTS rewards  (
                run_id TEXT NOT NULL,
                reward DOUBLE NOT NULL,
            );
            """

BENCHMARK_RUN_SCHEMA = """
            CREATE TABLE IF NOT EXISTS benchmark_run (
                benchmark_run_id TEXT,
                timestamp TEXT NOT NULL,
                seed INTEGER NOT NULL,
                generator_name TEXT NOT NULL,
                generator_params JSON NOT NULL,

                PRIMARY KEY (benchmark_run_id)
            );
            """
SOLVER_CONFIG_SCHEMA = """
            CREATE TABLE IF NOT EXISTS solver_config (
                solver_id TEXT NOT NULL,
                solver_type TEXT NOT NULL,
                config_params JSON NOT NULL,
                PRIMARY KEY (solver_id)
            );
            """

SOLVER_LOCKS_SCHEMA = """
            CREATE TABLE IF NOT EXISTS solver_locks (
                solver_id TEXT NOT NULL,
                worker_id TEXT NOT NULL,
                lock_time TIMESTAMP NOT NULL,
                PRIMARY KEY (solver_id)
            );
            """
