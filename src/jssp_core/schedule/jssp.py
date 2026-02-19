from collections import defaultdict
from collections.abc import Generator

import numpy as np

from jssp_core.models.operation import OperationInfo, ScheduledOperation


class Schedule:
    """
    Core JSSP scheduling class that handles the state and logic of job shop scheduling.

    This class maintains the state of a JSSP solution, including which operations
    have been scheduled, their start times, and the availability of jobs and machines.
    It is designed to be framework-agnostic and can be used standalone or within
    different environment interfaces.

    Attributes:
        instance (list[list[tuple[int, int]]]): The JSSP instance data.
        num_jobs (int): Number of jobs in the instance.
        num_operations (int): Total number of operations across all jobs.
        num_machines (int): Total number of machines in the instance.
        scheduled (dict[tuple[int, int], float]): Mapping of (job_id, op_id) to start_time.
        job_next_op (list[int]): Index of the next operation to be scheduled for each job.
        job_ready_time (list[float]): Earliest time each job is ready for its next operation.
        machine_ready_time (defaultdict[int, float]): Earliest time each machine is available.
        eligible_operations (dict[tuple[int, int], int]): Binary flags for operation eligibility.
    """

    def __init__(self, instance: list[list[tuple[int, int]]]):
        """
        Initialize a new Schedule with a JSSP instance.

        Args:
            instance: JSSP instance data where each job is a list of
                     (machine_id, duration) tuples.
        """
        self.instance = instance
        self.num_jobs = len(instance)
        self.num_operations = sum(len(job) for job in instance)
        self.num_machines = max(m for job in instance for m, _ in job) + 1
        self.scheduled: dict[tuple[int, int], float] = {}
        self.job_next_op: list[int] = []
        self.job_ready_time: list[float] = []
        self.machine_ready_time: defaultdict[int, float] = defaultdict(float)
        self.eligible_operations: dict[tuple[int, int], int] = {}

        # Core scheduling state
        self.reset()

    def reset(self) -> None:
        """
        Reset the schedule to its initial state.

        Clears all scheduled operations and resets job/machine availability times.
        """
        self.job_next_op = [0] * self.num_jobs
        self.job_ready_time = [0.0] * self.num_jobs
        self.machine_ready_time = defaultdict(float)
        self.scheduled = {}  # (job_id, op_id) -> start_time

        # Operation eligibility tracking
        self.eligible_operations = {
            (j, i): int(i == 0)
            for j in range(self.num_jobs)
            for i in range(len(self.instance[j]))  # Number of Operations for job j
        }

    def get_all_scheduled_operations(self) -> Generator[ScheduledOperation, None, None]:
        """
        Generator yielding all currently scheduled operations.

        Yields:
            ScheduledOperation: Object containing operation info and timing.
        """
        for key, val in self.scheduled.items():
            j_id, o_id = key
            op = self.instance[j_id][o_id]
            yield ScheduledOperation(
                operation=OperationInfo(
                    job_id=j_id, op_id=o_id, machine=op[0], duration=op[1]
                ),
                start_time=int(val),
                end_time=int(val + op[1]),
            )

    def get_eligible_jobs(self) -> list[int]:
        """
        Get IDs of jobs that have an operation ready to be scheduled.

        Returns:
            list[int]: List of eligible job IDs.
        """
        eligible = []
        for j in range(self.num_jobs):
            i = self.job_next_op[j]
            if i < len(self.instance[j]):
                if i == 0 or (j, i - 1) in self.scheduled:
                    eligible.append(j)
        return eligible

    def get_valid_job_mask(self) -> np.ndarray:
        """
        Get a binary mask indicating eligible jobs.

        Returns:
            np.ndarray: Array of shape (num_jobs,) where 1 indicates
                        the job can be scheduled next, 0 otherwise.
        """
        mask = np.zeros(self.num_jobs, dtype=np.int8)
        for job_id in range(self.num_jobs):
            if self.can_schedule_job(job_id):
                mask[job_id] = 1
        return mask

    def get_valid_operation_mask(self) -> np.ndarray:
        """
        Get a binary mask indicating which operations are valid to schedule.

        An operation is valid if it hasn't been scheduled yet and its
        preceding operation in the same job (if any) is already scheduled.

        Returns:
            np.ndarray: Boolean array of shape (num_operations,) where True
                        indicates the operation is valid to schedule.
        """
        mask = np.zeros(self.num_operations, dtype=bool)

        # Iterate through all operations in order
        op_index = 0
        for job_id in range(self.num_jobs):
            for op_id in range(len(self.instance[job_id])):
                # Check if this operation is eligible
                if self.eligible_operations.get((job_id, op_id), 0) == 1:
                    mask[op_index] = True
                op_index += 1

        return mask

    def flat_index_to_job_op(self, flat_index: int) -> tuple[int, int]:
        """
        Convert a flattened operation index to (job_id, op_id) tuple.

        Args:
            flat_index: Flattened index in range [0, num_operations)

        Returns:
            Tuple[int, int]: (job_id, op_id) corresponding to the flat index

        Raises:
            ValueError: If flat_index is out of bounds
        """
        if flat_index < 0 or flat_index >= self.num_operations:
            raise ValueError(
                f"flat_index {flat_index} out of bounds [0, {self.num_operations})"
            )

        current_index = 0
        for job_id in range(self.num_jobs):
            job_length = len(self.instance[job_id])
            if current_index + job_length > flat_index:
                op_id = flat_index - current_index
                return (job_id, op_id)
            current_index += job_length

        # This should never happen if flat_index is within bounds
        raise RuntimeError(
            f"Failed to convert flat_index {flat_index} to (job_id, op_id)"
        )

    def job_op_to_flat_index(self, job_id: int, op_id: int) -> int:
        """
        Convert (job_id, op_id) tuple to flattened operation index.

        Args:
            job_id: Job identifier
            op_id: Operation identifier within the job

        Returns:
            int: Flattened index corresponding to the operation

        Raises:
            ValueError: If job_id or op_id is out of bounds
        """
        if job_id < 0 or job_id >= self.num_jobs:
            raise ValueError(f"job_id {job_id} out of bounds [0, {self.num_jobs})")

        if op_id < 0 or op_id >= len(self.instance[job_id]):
            raise ValueError(
                f"op_id {op_id} out of bounds [0, {len(self.instance[job_id])})"
            )

        flat_index = 0
        for j in range(job_id):
            flat_index += len(self.instance[j])
        flat_index += op_id

        return flat_index

    def is_job_schedulable_on_machine(self, job_id: int, machine_id: int) -> bool:
        """
        Determines if the specified job can be scheduled on the given machine.

        Args:
            job_id: The ID of the job to check.
            machine_id: The ID of the machine to check against.

        Returns:
            bool: False if the job can be scheduled on the specified machine, True otherwise.
        """

        if not self.can_schedule_job(job_id):
            return True

        op_id = self.job_next_op[job_id]
        machine, _ = self.instance[job_id][op_id]
        if machine != machine_id:
            return True
        return False

    def get_operation_info(self, job_id: int, op_id: int) -> OperationInfo:
        """Get information about a specific operation"""
        machine, duration = self.instance[job_id][op_id]
        return OperationInfo(job_id, op_id, machine, duration)

    def get_next_operation_info(self, job_id: int) -> OperationInfo | None:
        """Get information about the next operation for a job"""
        op_id = self.job_next_op[job_id]
        if op_id < len(self.instance[job_id]):
            return self.get_operation_info(job_id, op_id)
        return None

    def get_current_operation_info(self, job_id: int) -> OperationInfo | None:
        """Get information about the current operation for a job"""
        op_id = self.job_next_op[job_id] - 1
        if op_id >= 0 and op_id < len(self.instance[job_id]):
            return self.get_operation_info(job_id, op_id)
        return None

    def can_schedule_job(self, job_id: int) -> bool:
        """Check if a job can be scheduled (has eligible operations)"""
        return job_id in self.get_eligible_jobs()

    def get_earliest_start_time(self, job_id: int, op_id: int) -> float:
        """Calculate the earliest possible start time for an operation"""
        machine, _ = self.instance[job_id][op_id]
        return max(
            self.job_ready_time[job_id], self.machine_ready_time.get(machine, 0.0)
        )

    def add_operation(self, job_id: int, op_id: int, start_time: float) -> bool:
        """Backward-compatible method to add an operation to the schedule"""
        return self.schedule_job(job_id)

    def get_operations_on_machine(self, machine_id: int) -> list[tuple[int, int]]:
        """
        Get all operations scheduled on a specific machine

        Args:
            machine_id: ID of the machine to query

        Returns:
            List of [(job_id, op_id), ...] scheduled on the machine
        """
        if machine_id < 0 or machine_id >= self.num_machines:
            raise ValueError(
                f"Invalid machine_id {machine_id}. Must be in range [0, {self.num_machines})."
            )

        ops_on_machine = []
        for job_id, job in enumerate(self.instance):
            for op_id, (m_id, _) in enumerate(job):
                if m_id == machine_id:
                    ops_on_machine.append((job_id, op_id))
        return ops_on_machine

    def schedule_job(self, job_id: int) -> bool:
        """
        Schedule the next operation for a given job

        Args:
            job_id: ID of the job to schedule

        Returns:
            True if operation was scheduled successfully, False otherwise
        """
        assert isinstance(job_id, (int, np.integer)), "job_id must be an integer"

        if not self.can_schedule_job(job_id):
            return False

        op_id = self.job_next_op[job_id]
        if op_id >= len(self.instance[job_id]):
            return False

        machine, duration = self.instance[job_id][op_id]
        start_time = self.get_earliest_start_time(job_id, op_id)
        end_time = start_time + duration

        # Update schedule
        self.scheduled[(job_id, op_id)] = start_time
        self.job_next_op[job_id] += 1
        self.job_ready_time[job_id] = float(end_time)
        self.machine_ready_time[machine] = float(end_time)

        # Update eligibility
        self.eligible_operations[(job_id, op_id)] = 0
        if op_id + 1 < len(self.instance[job_id]):
            self.eligible_operations[(job_id, op_id + 1)] = 1

        return True

    def get_eligible_operations(self) -> dict[tuple[int, int], int]:
        """Get all operations currently eligible for scheduling."""
        eligible_dict = {}
        for k, v in self.eligible_operations.items():
            if v == 1:
                eligible_dict[k] = v

        return eligible_dict

    def is_complete(self) -> bool:
        """Check if all operations have been scheduled"""
        return len(self.scheduled) == self.num_operations

    def get_makespan(self) -> float:
        """Get the current makespan (maximum completion time)"""
        if not self.scheduled:
            return 0.0
        return max(self.job_ready_time)

    def get_scheduled_operations_count(self) -> int:
        """Get the number of operations currently scheduled"""
        return len(self.scheduled)

    def get_remaining_operations_count(self) -> int:
        """Get the number of operations remaining to be scheduled"""
        return self.num_operations - len(self.scheduled)

    def get_scheduled_operations_list(self) -> list[ScheduledOperation]:
        """
        Get all scheduled operations as ScheduledOperation objects

        Returns:
            List of ScheduledOperation objects
        """
        from jssp_core.models.operation import OperationInfo, ScheduledOperation

        operations = []
        for (job_id, op_id), start_time in self.scheduled.items():
            machine, duration = self.instance[job_id][op_id]
            operation = OperationInfo(
                job_id=job_id, op_id=op_id, machine=machine, duration=duration
            )
            scheduled_op = ScheduledOperation(
                operation=operation,
                start_time=int(start_time),
                end_time=int(start_time + duration),
            )
            operations.append(scheduled_op)
        return operations

    def estimate_completion_time(self, heuristic: str = "SPT") -> float:
        """
        Estimate the total completion time using any heuristic from heuristics.py
        This doesn't modify the current schedule state

        Args:
            heuristic: Which heuristic to use for estimation
                      ("SPT", "LPT", "MWR", "LWR", "FCFS", "Random")

        Returns:
            Estimated completion time (makespan)
        """

        # Lazy import to avoid circular dependency
        from jssp_core.solver.heuristic_solver import JSSPHeuristicSolver

        # Create a partial solution from the current schedule state
        partial_solution = Schedule(self.instance)

        # Add all currently scheduled operations
        for (job_id, op_id), start_time in self.scheduled.items():
            partial_solution.add_operation(job_id, op_id, start_time)

        # Use heuristic solver to complete the schedule from current state
        solver = JSSPHeuristicSolver(self.instance)
        complete_solution = solver.continue_from_partial_solution(
            partial_solution, heuristic
        )

        return float(complete_solution.get_makespan())

    def estimate_completion_time_multiple_heuristics(
        self, heuristics: list[str] | None = None
    ) -> dict[str, float]:
        """
        Estimate completion time using multiple heuristics and return all results

        Args:
            heuristics: List of heuristics to try. If None, uses common ones.

        Returns:
            Dictionary mapping heuristic name to estimated completion time
        """
        if heuristics is None:
            heuristics = ["SPT", "LPT", "MWR", "LWR", "FCFS"]

        results = {}
        for heuristic in heuristics:
            try:
                results[heuristic] = self.estimate_completion_time(heuristic)
            except Exception:
                # Skip heuristics that fail (e.g., if they need additional parameters)
                results[heuristic] = float("inf")

        return results

    def get_best_heuristic_estimate(
        self, heuristics: list[str] | None = None
    ) -> tuple[str, float]:
        """
        Get the best (lowest) completion time estimate among multiple heuristics

        Args:
            heuristics: List of heuristics to try. If None, uses common ones.

        Returns:
            Tuple of (best_heuristic_name, best_estimated_time)
        """
        estimates = self.estimate_completion_time_multiple_heuristics(heuristics)
        best_heuristic = min(estimates.items(), key=lambda x: x[1])
        return best_heuristic

    def get_job_progress(self, job_id: int) -> tuple[int, int]:
        """Get progress of a job (completed_operations, total_operations)"""
        return self.job_next_op[job_id], len(self.instance[job_id])

    def get_machine_utilization(self) -> dict[int, float]:
        """Get current utilization (busy time) for each machine"""

        utilization = {}
        for machine_id in range(self.num_machines):
            utilization[machine_id] = self.machine_ready_time.get(machine_id, 0.0)
        return utilization

    def get_schedule_summary(self, heuristic: str = "SPT") -> dict:
        """Get a summary of the current schedule state"""
        return {
            "scheduled_operations": len(self.scheduled),
            "total_operations": self.num_operations,
            "completion_percentage": (len(self.scheduled) / self.num_operations) * 100,
            "current_makespan": self.get_makespan(),
            "eligible_jobs": self.get_eligible_jobs(),
            "is_complete": self.is_complete(),
        }

    def build_precedence_edge_index(
        self, self_loop: bool = False, both_directions: bool = False
    ) -> np.ndarray:
        """
        Build edge index matrix for precedence constraints within jobs.

        Args:
            self_loop: If True, adds self-loop edges for each node
            both_directions: If True, adds reverse edges (target -> source) for each precedence edge

        Returns:
            np.ndarray: Edge index array of shape (2, num_edges) representing
                       directed edges from operation i to operation i+1 in each job
        """
        # Create operation to node mapping
        op_node_id = {}
        node_id = 0
        for job_idx, job in enumerate(self.instance):
            for op_idx in range(len(job)):
                op_node_id[(job_idx, op_idx)] = node_id
                node_id += 1

        edge_sources = []
        edge_targets = []

        for job_idx, job in enumerate(self.instance):
            # Add precedence edges within each job
            for op_idx in range(len(job) - 1):
                source_node = op_node_id[(job_idx, op_idx)]
                target_node = op_node_id[(job_idx, op_idx + 1)]

                # Add forward edge (source -> target)
                edge_sources.append(source_node)
                edge_targets.append(target_node)

                # Add reverse edge (target -> source) if both_directions is True
                if both_directions:
                    edge_sources.append(target_node)
                    edge_targets.append(source_node)

        # Add self-loop edges if requested
        if self_loop:
            for node_id in range(self.num_operations):
                edge_sources.append(node_id)
                edge_targets.append(node_id)

        # Convert to edge_index format: [2, num_edges]
        if edge_sources:
            edge_index = np.array([edge_sources, edge_targets], dtype=np.int64)
        else:
            # Handle case with no edges (single operation jobs)
            edge_index = np.zeros((2, 0), dtype=np.int64)

        return edge_index

    def build_machine_edge_index(self, self_loop: bool = False) -> np.ndarray:
        """
        Build edge index matrix for machine constraints.

        Creates edges between operations that use the same machine.
        This represents machine capacity constraints where operations on the same
        machine cannot be executed simultaneously.

        Args:
            self_loop: If True, adds self-loop edges for each node

        Returns:
            np.ndarray: Edge index array of shape (2, num_edges) representing
                       undirected edges between operations that share the same machine

        Notes:
            - Fully connected edges num = n*(n-1)/2 for n operations on the same machine
            - So all machine edges = sum(n*(n-1)/2) over all machines (* 2 for both directions)
        """
        # Create operation to node mapping
        op_node_id = {}
        node_id = 0
        for job_idx, job in enumerate(self.instance):
            for op_idx in range(len(job)):
                op_node_id[(job_idx, op_idx)] = node_id
                node_id += 1

        # Group operations by machine
        machine_operations = {}
        for job_idx, job in enumerate(self.instance):
            for op_idx, (machine, duration) in enumerate(job):
                if machine not in machine_operations:
                    machine_operations[machine] = []
                machine_operations[machine].append((job_idx, op_idx))

        edge_sources = []
        edge_targets = []

        # Create edges between operations on the same machine
        for machine, operations in machine_operations.items():
            # Create all pairwise edges (undirected graph)
            for i in range(len(operations)):
                for j in range(i + 1, len(operations)):
                    job_i, op_i = operations[i]
                    job_j, op_j = operations[j]

                    node_i = op_node_id[(job_i, op_i)]
                    node_j = op_node_id[(job_j, op_j)]

                    # # Add both directions for undirected edge
                    edge_sources.extend([node_i, node_j])
                    edge_targets.extend([node_j, node_i])

        # Add self-loop edges if requested
        if self_loop:
            for node_id in range(self.num_operations):
                edge_sources.append(node_id)
                edge_targets.append(node_id)

        # Convert to edge_index format: [2, num_edges]
        if edge_sources:
            edge_index = np.array([edge_sources, edge_targets], dtype=np.int64)
        else:
            # Handle case with no edges (no operations share machines)
            edge_index = np.zeros((2, 0), dtype=np.int64)

        return edge_index

    def build_machine_precedence_edge_index(
        self, self_loop: bool = False
    ) -> np.ndarray:
        """
        Build edge index matrix for machine precedence based on current schedule.

        Creates directed edges between operations on the same machine based on their
        scheduled order. Each operation connects to its predecessor and successor
        on the same machine, representing the temporal ordering constraints.

        Args:
            self_loop: If True, adds self-loop edges for each node

        Returns:
            np.ndarray: Edge index array of shape (2, num_edges) representing
                       directed edges from predecessor to successor on same machine
        """
        if not self.scheduled:
            # No operations scheduled yet, return empty edge index
            if self_loop:
                edge_sources = list(range(self.num_operations))
                edge_targets = list(range(self.num_operations))
                return np.array([edge_sources, edge_targets], dtype=np.int64)
            else:
                return np.zeros((2, 0), dtype=np.int64)

        # Create operation to node mapping
        op_node_id = {}
        node_id = 0
        for job_idx, job in enumerate(self.instance):
            for op_idx in range(len(job)):
                op_node_id[(job_idx, op_idx)] = node_id
                node_id += 1

        # Group scheduled operations by machine with their start times
        machine_operations = {}
        for (job_id, op_id), start_time in self.scheduled.items():
            machine, duration = self.instance[job_id][op_id]
            if machine not in machine_operations:
                machine_operations[machine] = []
            machine_operations[machine].append((start_time, job_id, op_id))

        edge_sources = []
        edge_targets = []

        # Create precedence edges for each machine
        for machine, operations in machine_operations.items():
            # Sort operations by start time to get the correct precedence order
            operations.sort(key=lambda x: x[0])  # Sort by start_time

            # Create edges from each operation to the next one on same machine
            for i in range(len(operations) - 1):
                _, job_i, op_i = operations[i]
                _, job_j, op_j = operations[i + 1]

                source_node = op_node_id[(job_i, op_i)]
                target_node = op_node_id[(job_j, op_j)]

                # Add directed edge from predecessor to successor
                edge_sources.append(source_node)
                edge_targets.append(target_node)

                # Add reverse edge (successor to predecessor)
                edge_sources.append(target_node)
                edge_targets.append(source_node)

        # Add self-loop edges if requested
        if self_loop:
            for node_id in range(self.num_operations):
                edge_sources.append(node_id)
                edge_targets.append(node_id)

        # Convert to edge_index format: [2, num_edges]
        if edge_sources:
            edge_index = np.array([edge_sources, edge_targets], dtype=np.int64)
        else:
            # Handle case with no edges
            if self_loop:
                edge_sources = list(range(self.num_operations))
                edge_targets = list(range(self.num_operations))
                edge_index = np.array([edge_sources, edge_targets], dtype=np.int64)
            else:
                edge_index = np.zeros((2, 0), dtype=np.int64)

        return edge_index

    def is_valid(self) -> bool:
        """Check if the current schedule is valid (no constraint violations)"""
        is_valid, violations = self.validate_schedule()
        return is_valid

    def validate_schedule(self) -> tuple[bool, list[str]]:
        """
        Validate the current schedule for constraint violations

        Returns:
            Tuple of (is_valid, list_of_violations)
        """
        violations = []

        # Check job precedence constraints
        for job_id in range(self.num_jobs):
            job_ops = [
                (op_id, start_time)
                for (j, op_id), start_time in self.scheduled.items()
                if j == job_id
            ]
            job_ops.sort(key=lambda x: x[0])  # Sort by operation ID

            for i in range(len(job_ops) - 1):
                op_id, start_time = job_ops[i]
                next_op_id, next_start_time = job_ops[i + 1]

                # Check if current operation finishes before next starts
                machine, duration = self.instance[job_id][op_id]
                end_time = start_time + duration

                if end_time > next_start_time:
                    violations.append(
                        f"Job {job_id}: Operation {op_id} ends at {end_time} "
                        f"but operation {next_op_id} starts at {next_start_time}"
                    )

        # Check machine capacity constraints
        for machine_id in range(self.num_machines):
            machine_ops = []
            for (job_id, op_id), start_time in self.scheduled.items():
                machine, duration = self.instance[job_id][op_id]
                if machine == machine_id:
                    machine_ops.append(
                        (start_time, start_time + duration, job_id, op_id)
                    )

            machine_ops.sort(key=lambda x: x[0])  # Sort by start time

            for i in range(len(machine_ops) - 1):
                _, end_time, job_id, op_id = machine_ops[i]
                next_start_time, _, next_job_id, next_op_id = machine_ops[i + 1]

                if end_time > next_start_time:
                    violations.append(
                        f"Machine {machine_id}: Operation J{job_id}O{op_id} ends at {end_time} "
                        f"but J{next_job_id}O{next_op_id} starts at {next_start_time}"
                    )

        return len(violations) == 0, violations

    def copy(self) -> "Schedule":
        """Create a deep copy of the current schedule"""
        new_schedule = Schedule(self.instance)
        new_schedule.job_next_op = self.job_next_op[:]
        new_schedule.job_ready_time = self.job_ready_time[:]
        new_schedule.machine_ready_time = dict(self.machine_ready_time)
        new_schedule.scheduled = dict(self.scheduled)
        new_schedule.eligible_operations = dict(self.eligible_operations)
        return new_schedule

    def apply_action_sequence(self, action_sequence: list[int]) -> bool:
        """
        Apply a sequence of job actions to the schedule

        Args:
            action_sequence: List of job IDs to schedule in order

        Returns:
            True if all actions were applied successfully, False otherwise
        """
        for job_id in action_sequence:
            if not self.schedule_job(job_id):
                return False
        return True

    def get_lower_bound_makespan(self) -> float:
        """
        Calculate the lower bound makespan for the current schedule state.

        The lower bound is calculated as the maximum of:
        1. Job-based lower bound: For each job, current completion time + remaining processing time
        2. Machine-based lower bound: For each machine, current completion time + remaining processing time
        3. Critical path lower bound: For each unscheduled operation, the critical longest path

        Returns:
            float: Lower bound estimate for makespan
        """
        if self.is_complete():
            return self.get_makespan()

        lower_bound = 0.0

        # 1. Job-based lower bound
        for job_id in range(self.num_jobs):
            job_lower_bound = self.job_ready_time[job_id]

            # Add remaining processing time for this job
            for op_id in range(self.job_next_op[job_id], len(self.instance[job_id])):
                _, duration = self.instance[job_id][op_id]
                job_lower_bound += duration

            lower_bound = max(lower_bound, job_lower_bound)

        # 2. Machine-based lower bound
        machine_remaining_work = [0.0] * self.num_machines

        # Calculate remaining work for each machine
        for job_id in range(self.num_jobs):
            for op_id in range(self.job_next_op[job_id], len(self.instance[job_id])):
                machine, duration = self.instance[job_id][op_id]
                machine_remaining_work[machine] += duration

        for machine_id in range(self.num_machines):
            machine_ready_time = self.machine_ready_time.get(machine_id, 0.0)
            machine_lower_bound = (
                machine_ready_time + machine_remaining_work[machine_id]
            )
            lower_bound = max(lower_bound, machine_lower_bound)

        # 3. Critical path lower bound for each unscheduled operation
        for job_id in range(self.num_jobs):
            for op_id in range(self.job_next_op[job_id], len(self.instance[job_id])):
                op_lower_bound = self._calculate_operation_critical_path_lower_bound(
                    job_id, op_id
                )
                lower_bound = max(lower_bound, op_lower_bound)

        return lower_bound

    def get_maximal_bound_makespan(self) -> float:
        """
        Calculate the maximal bound makespan for the current schedule state.

        The maximal bound is calculated as the sum of:
        - Current makespan
        - Total remaining processing time for all unscheduled operations

        Returns:
            float: Maximal bound estimate for makespan
        """
        if self.is_complete():
            return self.get_makespan()

        current_makespan = self.get_makespan()
        total_remaining_time = 0.0

        # Sum remaining processing time for all unscheduled operations
        for job_id in range(self.num_jobs):
            for op_id in range(self.job_next_op[job_id], len(self.instance[job_id])):
                _, duration = self.instance[job_id][op_id]
                total_remaining_time += duration

        maximal_bound = current_makespan + total_remaining_time
        return maximal_bound

    def _calculate_operation_critical_path_lower_bound(
        self, job_id: int, op_id: int
    ) -> float:
        """
        Calculate the critical path lower bound for a specific operation.

        This includes:
        - Time to reach this operation (job precedence)
        - Time for the operation itself
        - Remaining work after this operation

        Args:
            job_id: Job ID
            op_id: Operation ID within the job

        Returns:
            float: Critical path lower bound for this operation
        """
        machine, duration = self.instance[job_id][op_id]

        # Time when this operation can start (considering job precedence)
        job_completion_before_op = self.job_ready_time[job_id]

        # Time when this operation can start (considering machine availability)
        machine_ready_time = self.machine_ready_time.get(machine, 0.0)

        # Earliest start time for this operation
        earliest_start = max(job_completion_before_op, machine_ready_time)

        # End time of this operation
        operation_end = earliest_start + duration

        # Remaining work in this job after this operation
        remaining_job_work = sum(
            self.instance[job_id][future_op][1]  # duration
            for future_op in range(op_id + 1, len(self.instance[job_id]))
        )

        # Critical path estimate: operation end + remaining work
        return operation_end + remaining_job_work

    def get_action_sequence(self) -> list[int]:
        """Get the sequence of actions (job_id, op_id) in the order they were scheduled"""
        # Sort scheduled operations by their start time
        sorted_operations = sorted(
            self.scheduled.items(), key=lambda x: x[1]
        )  # ( (job_id, op_id), start_time )

        # Extract job IDs in the order of scheduling
        action_sequence = [job_id for (job_id, op_id), start_time in sorted_operations]

        return action_sequence

    def get_operation_lower_bounds(self) -> dict[tuple[int, int], float]:
        """
        Calculate the lower bound per operation by cumulating job constraints.

        For each operation, the lower bound is calculated as:
        - If previous operations in the same job are scheduled: max(last_scheduled_completion, cumulative_duration)
        - If previous operations are not scheduled: cumulative duration from job start

        Example:
        - Operation J0O0 (duration=3): lower_bound = 3
        - Operation J0O1 (duration=2): lower_bound = 5 (3+2)
        - If J0O0 was scheduled and finished at 10: J0O1 lower_bound = 12 (max(10, 0) + 2)

        Returns:
            Dictionary mapping (job_id, op_id) to lower bound time
        """
        lower_bounds = {}

        for job_id in range(self.num_jobs):
            cumulative_duration = 0
            last_completion_time = 0

            for op_id in range(len(self.instance[job_id])):
                machine, duration = self.instance[job_id][op_id]

                if (job_id, op_id) in self.scheduled:
                    # Operation is already scheduled - use actual completion time
                    start_time = self.scheduled[(job_id, op_id)]
                    completion_time = start_time + duration
                    lower_bounds[(job_id, op_id)] = completion_time
                    last_completion_time = completion_time
                    cumulative_duration = 0  # Reset since we have actual timing info
                else:
                    # Operation not yet scheduled - use cumulative approach
                    cumulative_duration += duration

                    if last_completion_time > 0:
                        # Previous operations were scheduled, use their completion time as base
                        lower_bounds[(job_id, op_id)] = (
                            last_completion_time + cumulative_duration
                        )
                    else:
                        # No previous operations scheduled, use pure cumulative duration
                        lower_bounds[(job_id, op_id)] = cumulative_duration

        return lower_bounds

    def get_operation_lower_bound(self, job_id: int, op_id: int) -> float:
        """
        Get the lower bound for a specific operation.

        Args:
            job_id: Job ID
            op_id: Operation ID within the job

        Returns:
            Lower bound time for the specified operation
        """
        if job_id < 0 or job_id >= self.num_jobs:
            raise ValueError(f"Invalid job_id: {job_id}")
        if op_id < 0 or op_id >= len(self.instance[job_id]):
            raise ValueError(f"Invalid op_id: {op_id} for job {job_id}")

        # Calculate lower bounds for all operations in this job up to the requested one
        cumulative_duration = 0
        last_completion_time = 0

        for current_op_id in range(op_id + 1):
            machine, duration = self.instance[job_id][current_op_id]

            if (job_id, current_op_id) in self.scheduled:
                # Operation is already scheduled - use actual completion time
                start_time = self.scheduled[(job_id, current_op_id)]
                completion_time = start_time + duration
                if current_op_id == op_id:
                    return completion_time
                last_completion_time = completion_time
                cumulative_duration = 0  # Reset since we have actual timing info
            else:
                # Operation not yet scheduled - add to cumulative duration
                cumulative_duration += duration

                if current_op_id == op_id:
                    if last_completion_time > 0:
                        # Previous operations were scheduled, use their completion time as base
                        return last_completion_time + cumulative_duration
                    else:
                        # No previous operations scheduled, use pure cumulative duration
                        return cumulative_duration

        # This should never be reached, but just in case
        return cumulative_duration

    def get_gantt_data(self) -> list[dict]:
        """
        Get data for Gantt chart visualization

        Returns:
            List of operation dictionaries with timing and machine info
        """
        gantt_data = []
        for (job_id, op_id), start_time in self.scheduled.items():
            machine, duration = self.instance[job_id][op_id]
            gantt_data.append(
                {
                    "job_id": job_id,
                    "op_id": op_id,
                    "machine": machine,
                    "start_time": start_time,
                    "duration": duration,
                    "end_time": start_time + duration,
                }
            )
        return gantt_data


if __name__ == "__main__":
    # Example usage and simple test
    instance = [
        [(0, 3), (1, 2), (2, 2)],
        [(0, 2), (2, 1), (1, 4)],
        [(1, 4), (2, 3)],
    ]

    schedule = Schedule(instance)

    print("Initial Schedule Summary:", schedule.get_schedule_summary())

    action_sequence = [0, 1, 2, 0, 1, 2, 0, 1]
    for job_id in action_sequence:
        success = schedule.schedule_job(job_id)
        if not success:
            print(f"Failed to schedule job {job_id}")
            break

    print("Final Schedule Summary:", schedule.get_schedule_summary())
    print("Scheduled Operations:", schedule.get_scheduled_operations_list())
    print("Is Schedule Valid?", schedule.is_valid())
    print("Makespan:", schedule.get_makespan())
    print("Lower Bound Makespan:", schedule.get_lower_bound_makespan())
    print("Operation Lower Bounds:", schedule.get_operation_lower_bounds())
    print("Action Sequence:", schedule.get_action_sequence())
    print("Precedence Edge Index:\n", schedule.build_precedence_edge_index())
    print("Machine Edge Index:\n", schedule.build_machine_edge_index())
    print(
        "Machine Precedence Edge Index:\n",
        schedule.build_machine_precedence_edge_index(),
    )
