import hashlib
import time
from collections import defaultdict


# Check OR-Tools availability locally to avoid circular import
try:
    from ortools.sat.python import cp_model

    ORTOOLS_AVAILABLE = True
except ImportError:
    cp_model = None
    ORTOOLS_AVAILABLE = False

from jssp_core.instances import JSSPInstance
from jssp_core.schedule import Schedule
from jssp_core.solver.base import SolverProtocol


class JSSPOptimalSolver(SolverProtocol):
    """
    Optimal solver for JSSP using constraint programming (Google OR-Tools)

    This solver finds the optimal solution by formulating JSSP as a constraint satisfaction problem
    and using CP-SAT solver. Suitable for small to medium-sized instances.
    """

    def __init__(self, time_limit_seconds: int = 300, timeout: int | None = None):
        if not ORTOOLS_AVAILABLE:
            raise ImportError(
                "Google OR-Tools is not installed. Install it with: pip install ortools"
            )

        if timeout is not None:
            time_limit_seconds = timeout

        self.time_limit_seconds = time_limit_seconds
        self.instance = None
        self._action_sequence: list[int] | None = None
        self._action_index: int = 0
        self._instance_hash: str | None = None

    def set_instance(self, instance):
        self.instance = instance
        self.num_jobs = len(instance)
        self.num_machines = max(m for job in instance for m, _ in job) + 1
        self.num_operations = sum(len(job) for job in instance)

    def solve(self, instance: JSSPInstance) -> Schedule:
        """
        Solve the JSSP instance optimally.

        Args:
            instance: The JSSP instance to solve.

        Returns:
            The optimal schedule.

        Raises:
            RuntimeError: If no solution is found within the time limit.
        """
        result, _ = self.solve_optimal(
            instance=instance, time_limit_seconds=self.time_limit_seconds, verbose=False
        )
        if result is None:
            raise RuntimeError(
                f"No solution found within time limit of {self.time_limit_seconds} seconds"
            )
        return result

    def step(self, current_schedule: Schedule, *args, **kwargs) -> int:
        """

        Step function required by SolverProtocol.

        The first call solves the full instance and caches the resulting action
        sequence. Subsequent calls return the next job_id from that sequence.
        """
        instance_hash = self._compute_instance_hash(current_schedule.instance)

        if self._action_sequence is None or instance_hash != self._instance_hash:
            solved_schedule = self.solve(current_schedule.instance)
            self._action_sequence = solved_schedule.get_action_sequence()
            self._action_index = 0
            self._instance_hash = instance_hash

        if self._action_index >= len(self._action_sequence):
            raise StopIteration("All actions have been yielded for this instance")

        job_id = self._action_sequence[self._action_index]
        self._action_index += 1
        return job_id

    def get_config_hash(self) -> str:
        """Return a hash of the solver configuration."""
        config = f"JSSPOptimalSolver_timeout={self.time_limit_seconds}"
        return hashlib.md5(config.encode()).hexdigest()

    def _compute_instance_hash(self, instance: JSSPInstance) -> str:
        """Stable hash for the current instance to reset cached actions when it changes."""
        return hashlib.md5(repr(instance).encode()).hexdigest()

    def _get_status_string(self, status: int) -> str:
        """Convert CP model status to string."""
        status_map = {
            cp_model.UNKNOWN: "UNKNOWN",
            cp_model.MODEL_INVALID: "MODEL_INVALID",
            cp_model.FEASIBLE: "FEASIBLE",
            cp_model.INFEASIBLE: "INFEASIBLE",
            cp_model.OPTIMAL: "OPTIMAL",
        }
        return status_map.get(status, "UNKNOWN")

    def _build_model(self, horizon: int):
        """Build the CP model for JSSP."""
        model = cp_model.CpModel()
        start_times = {}
        end_times = {}
        intervals = {}

        # Create variables for each operation
        for job_id in range(self.num_jobs):
            for op_id in range(len(self.instance[job_id])):
                machine, duration = self.instance[job_id][op_id]

                # Start time variable
                start_var = model.NewIntVar(0, horizon, f"start_j{job_id}_o{op_id}")
                start_times[(job_id, op_id)] = start_var

                # End time variable
                end_var = model.NewIntVar(0, horizon, f"end_j{job_id}_o{op_id}")
                end_times[(job_id, op_id)] = end_var

                # Interval variable (connects start, duration, end)
                interval_var = model.NewIntervalVar(
                    start_var, duration, end_var, f"interval_j{job_id}_o{op_id}"
                )
                intervals[(job_id, op_id)] = interval_var

        # Precedence constraints: operations within same job must be sequential
        for job_id in range(self.num_jobs):
            for op_id in range(len(self.instance[job_id]) - 1):
                model.Add(
                    start_times[(job_id, op_id + 1)] >= end_times[(job_id, op_id)]
                )

        # Machine capacity constraints: no two operations on same machine can overlap
        machine_operations = defaultdict(list)
        for job_id in range(self.num_jobs):
            for op_id in range(len(self.instance[job_id])):
                machine, _ = self.instance[job_id][op_id]
                machine_operations[machine].append(intervals[(job_id, op_id)])

        for machine_id in range(self.num_machines):
            if machine_operations[machine_id]:
                model.AddNoOverlap(machine_operations[machine_id])

        # Makespan variable (objective to minimize)
        makespan = model.NewIntVar(0, horizon, "makespan")

        # Makespan constraints: makespan >= end time of all operations
        for job_id in range(self.num_jobs):
            for op_id in range(len(self.instance[job_id])):
                model.Add(makespan >= end_times[(job_id, op_id)])

        # Objective: minimize makespan
        model.Minimize(makespan)

        return model, start_times, makespan

    def solve_optimal(
        self,
        time_limit_seconds: int = 300,
        verbose: bool = True,
        instance: Schedule | None = None,
    ) -> tuple[Schedule | None, int]:
        """
        Solve JSSP optimally using constraint programming

        Args:
            time_limit_seconds: Maximum time to spend solving (default 5 minutes)
            verbose: Whether to print progress information
            instance: Instance to solve

        Returns:
            Tuple of (Optimal Schedule if found within time limit or None, status code)
        """
        if instance is not None:
            self.set_instance(instance)

        if self.instance is None:
            raise ValueError("No instance provided to solve")

        if not ORTOOLS_AVAILABLE:
            raise ImportError("Google OR-Tools is required for optimal solving")

        # Calculate upper bound for makespan (sum of all processing times)
        horizon = sum(duration for job in self.instance for _, duration in job)

        if verbose:
            print(f"Setting up CP model with horizon {horizon}")
            print(
                f"Problem size: {self.num_jobs} jobs, {self.num_machines} machines, {self.num_operations} operations"
            )

        model, start_times, makespan = self._build_model(horizon)

        # Solve the model
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = time_limit_seconds

        if verbose:
            print(f"Solving with time limit of {time_limit_seconds} seconds...")
            start_time = time.time()

        status = solver.Solve(model)

        if verbose:
            solve_time = time.time() - start_time
            print(f"Solver finished in {solve_time:.2f} seconds")

        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            if verbose:
                status_str = self._get_status_string(status)
                print(
                    f"{status_str} solution found with makespan: {solver.Value(makespan)}"
                )
                if status == cp_model.FEASIBLE:
                    print("(May not be optimal due to time limit)")
            return self._extract_solution(solver, start_times), status
        else:
            if verbose:
                print("No solution found within time limit")
            return None, status

    def _extract_solution(self, solver, start_times) -> Schedule:
        """Extract a CP solution into a Schedule object."""
        solution = Schedule(self.instance)

        # Populate schedule fields directly from solver values
        solution.scheduled = {}
        job_completion = [0.0 for _ in range(self.num_jobs)]
        machine_completion = defaultdict(float)

        for job_id in range(self.num_jobs):
            for op_id in range(len(self.instance[job_id])):
                start_time = solver.Value(start_times[(job_id, op_id)])
                duration = self.instance[job_id][op_id][1]
                end_time = start_time + duration

                solution.scheduled[(job_id, op_id)] = float(start_time)
                job_completion[job_id] = max(job_completion[job_id], end_time)

                machine = self.instance[job_id][op_id][0]
                machine_completion[machine] = max(machine_completion[machine], end_time)

        # Mark all operations as completed
        solution.job_next_op = [len(job) for job in self.instance]
        solution.job_ready_time = job_completion
        solution.machine_ready_time = dict(machine_completion)

        # No further operations are eligible
        solution.eligible_operations = {
            (job_id, op_id): 0
            for job_id in range(self.num_jobs)
            for op_id in range(len(self.instance[job_id]))
        }

        return solution

    def solve_with_timeout(
        self, timeout_seconds: int = 60, verbose: bool = True
    ) -> tuple[Schedule | None, bool]:
        """
        Solve with timeout and return solution status

        Returns:
            Tuple of (solution, is_optimal)
        """
        solution, status = self.solve_optimal(timeout_seconds, verbose)

        if solution is None:
            return None, False

        # Check if solution is truly optimal (would need to compare with lower bound)
        # For now, assume it's optimal if found quickly
        return solution, status == cp_model.OPTIMAL

    def get_name(self) -> str:
        return "OR_CP_SAT"

    @property
    def name(self) -> str:
        return self.get_name()

    def get_type(self):
        from jssp_core.solver.base import SolverType

        return SolverType.OPTIMAL

    def solve_with_info(self, instance: JSSPInstance) -> "SolveOutput":
        from jssp_core.solver.base import SolveOutput

        result, status = self.solve_optimal(
            instance=instance, time_limit_seconds=self.time_limit_seconds, verbose=False
        )

        # If no solution found, we might want to return an empty schedule or raise error
        # solve_optimal returns None if no solution found
        if result is None:
            # For now, let's raise error as solve() does
            raise RuntimeError(
                f"No solution found within time limit of {self.time_limit_seconds} seconds"
            )

        return SolveOutput(
            solution=result, info={"cp_model_status": self._get_status_string(status)}
        )


if __name__ == "__main__":
    from jssp_core.instances import get_instance

    instance = get_instance({"type": "path", "path": "jssp_instances/ft20"})
    solver = JSSPOptimalSolver(time_limit_seconds=1)
    output = solver.solve_with_info(instance)
    print(output)
