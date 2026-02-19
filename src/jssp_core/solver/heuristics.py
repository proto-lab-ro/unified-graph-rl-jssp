import random

from jssp_core.domain.domains import ItemDataType
from jssp_core.instances import JSSPInstance
from jssp_core.schedule import Schedule
from jssp_core.solver.base import Heuristic


heuristic_rng = random.Random(42)


class ScheduleProxy:
    """
    A temporary view of the schedule that 'lies' about which operations are eligible.
    It passes all other attribute access requests directly to the real schedule.
    """

    def __init__(self, real_schedule, filtered_ops):
        self._real_schedule = real_schedule
        self._filtered_ops = filtered_ops

    def get_eligible_operations(self):
        """Override to return only our non-delay filtered operations"""
        return self._filtered_ops

    def __getattr__(self, name):
        """Pass any other method calls (get_earliest_start_time, instance, etc.) to the real schedule"""
        return getattr(self._real_schedule, name)


class NonDelay(Heuristic):
    """
    A generic wrapper that turns ANY heuristic into a Non-Delay heuristic.

    Logic:
    1. Look at all currently eligible jobs.
    2. Find the global minimum earliest start time (t_min).
    3. Filter the list to keep ONLY jobs that start at t_min.
    4. Pass this filtered list to the base heuristic to make the final choice.
    """

    def __init__(self, base_heuristic: Heuristic):
        self.base_heuristic = base_heuristic
        self.item_type = base_heuristic.item_type

    def step(self, current_schedule) -> int:
        # 1. Get the "true" eligible operations
        all_eligible = current_schedule.get_eligible_operations()

        if not all_eligible:
            raise ValueError("No eligible operations")

        # 2. Calculate start times for all candidates
        # Store as list of tuples: (job_id, op_id, start_time, original_value)
        candidates = []
        min_start_time = float("inf")

        for (job_id, op_id), val in all_eligible.items():
            start_time = current_schedule.get_earliest_start_time(job_id, op_id)
            if start_time < min_start_time:
                min_start_time = start_time
            candidates.append((job_id, op_id, start_time, val))

        # 3. Filter: Keep only operations that can start NOW (at min_start_time)
        # Reconstruct the dictionary format expected by the base heuristics
        non_delay_ops = {
            (j, o): v
            for j, o, t, v in candidates
            if t
            == min_start_time  # float comparison warning: in complex cases use abs(t - min) < 1e-6
        }

        # 4. Create the Proxy
        # This proxy acts exactly like 'current_schedule', but when asked for
        # eligible operations, it only reveals the non-delay ones.
        proxy_schedule = ScheduleProxy(current_schedule, non_delay_ops)

        # 5. Let the base heuristic decide among the non-delay options
        return self.base_heuristic.step(proxy_schedule)


class RandomStepHeuristic(Heuristic):
    """
    A wrapper that picks a random action every nth step.
    """

    def __init__(self, base_heuristic: Heuristic, n: int):
        self.base_heuristic = base_heuristic
        self.n = n
        self.step_count = 0
        self.item_type = base_heuristic.item_type

    def step(self, current_schedule) -> int:
        self.step_count += 1
        if self.step_count % self.n == 0:
            # Random action
            eligible_ops = current_schedule.get_eligible_operations()
            if not eligible_ops:
                raise ValueError("No eligible operations")

            # eligible_ops keys are (job_id, op_id)
            random_key = random.choice(list(eligible_ops.keys()))
            return random_key[0]  # Return job_id

        return self.base_heuristic.step(current_schedule)

    def solve(self, instance: JSSPInstance) -> Schedule:
        self.step_count = 0
        return super().solve(instance)

    def reset_count(self):
        self.step_count = 0


# ===========================
# Heuristic Implementations for Jobs
# ===========================


class SPT(Heuristic):
    """
    Shortest Processing Time (SPT) heuristic
    Always schedules the operation with the shortest duration among available operations
    """

    item_type = ItemDataType.JOB

    def step(self, current_schedule) -> int:
        eligible_ops = current_schedule.get_eligible_operations()
        process_durations = {}
        for (job_id, op_id), _ in eligible_ops.items():
            machine, duration = current_schedule.instance[job_id][op_id]
            process_durations[(job_id, op_id)] = duration
            if not process_durations:
                raise ValueError("No eligible operations")

        min_key = min(process_durations, key=process_durations.get)

        job_key = min_key[0]
        return job_key  # Return job_id


class SMPT(Heuristic):
    """
    Shortest Machine Processing Time (SMPT) heuristic
    Always schedules the operation with the shortest duration among available operations
    """

    item_type = ItemDataType.JOB

    def step(self, current_schedule) -> int:
        eligible_ops = current_schedule.get_eligible_operations()
        least_duration_jobs = list()
        # Look through all machines
        for machine_id in range(current_schedule.num_machines):
            process_durations = {}
            # Check which jobs can be scheduled in the machine
            for (job_id, op_id), _ in eligible_ops.items():
                if current_schedule.is_job_schedulable_on_machine(job_id, machine_id):
                    _, duration = current_schedule.instance[job_id][op_id]
                    process_durations[(job_id, op_id)] = duration
            if not process_durations:
                continue
            min_key = min(process_durations, key=process_durations.get)
            least_duration_jobs.append(min_key)

        # Now select the job with the earliest starting time
        earliest_starting_time = float("inf")
        earliest_job_key = -1
        for min_key in least_duration_jobs:
            start_time = current_schedule.get_earliest_start_time(
                min_key[0], min_key[1]
            )
            if start_time < earliest_starting_time:
                earliest_job_key = min_key[0]
                earliest_starting_time = start_time

        return earliest_job_key


class LPT(Heuristic):
    """
    Longest Processing Time (LPT) heuristic
    Always schedules the operation with the longest duration among available operations
    """

    item_type = ItemDataType.JOB

    def step(self, current_schedule) -> int:
        eligible_ops = current_schedule.get_eligible_operations()
        process_durations = {}
        for (job_id, op_id), _ in eligible_ops.items():
            machine, duration = current_schedule.instance[job_id][op_id]
            process_durations[(job_id, op_id)] = duration
            if not process_durations:
                raise ValueError("No eligible operations")

        min_key = max(process_durations, key=process_durations.get)

        job_key = min_key[0]
        return job_key  # Return job_id


class MWR(Heuristic):
    """
    Most Work Remaining (MWR) heuristic
    Prioritizes jobs with the most total processing time remaining
    """

    item_type = ItemDataType.JOB

    def step(self, current_schedule) -> int:
        eligible_ops = current_schedule.get_eligible_operations()
        remaining_work_dict = {}
        for (job_id, op_id), _ in eligible_ops.items():
            # Calculate remaining work for this job
            remaining_work = sum(
                d for _, d in current_schedule.instance[job_id][op_id:]
            )
            remaining_work_dict[job_id] = remaining_work

        job_key = max(remaining_work_dict, key=remaining_work_dict.get)
        return job_key  # Return job_id


class LWR(Heuristic):
    """
    Least Work Remaining (LWR) heuristic
    Prioritizes jobs with the least total processing time remaining
    """

    item_type = ItemDataType.JOB

    def step(self, current_schedule) -> int:
        eligible_ops = current_schedule.get_eligible_operations()
        remaining_work_dict = {}
        for (job_id, op_id), _ in eligible_ops.items():
            # Calculate remaining work for this job
            remaining_work = sum(
                d for _, d in current_schedule.instance[job_id][op_id:]
            )
            remaining_work_dict[job_id] = remaining_work
        job_key = min(remaining_work_dict, key=remaining_work_dict.get)
        return job_key  # Return job_id


class FDDMWR(Heuristic):
    """
    Flow Due Date/ Most Work Remaining (FDD/MWR) Heuristic
    Based on MWR, also takes into account the "urgency" of the job
    """

    item_type = ItemDataType.JOB

    def step(self, current_schedule) -> int:
        eligible_ops = current_schedule.get_eligible_operations()

        fdd_mwr_ratios = {}
        for (job_id, op_id), _ in eligible_ops.items():
            earliest_start_time = current_schedule.get_earliest_start_time(
                job_id, op_id
            )
            # Calculate remaining work for this job
            remaining_work = sum(
                d for _, d in current_schedule.instance[job_id][op_id:]
            )

            fdd = earliest_start_time + remaining_work

            ratio = fdd / (remaining_work + 1e-9)

            fdd_mwr_ratios[job_id] = ratio

        job_key = min(fdd_mwr_ratios, key=fdd_mwr_ratios.get)
        return job_key


class MOR(Heuristic):
    """
    Most Operations Remaining (MOR) heuristic
    Prioritizes jobs with the most operations remaining
    """

    item_type = ItemDataType.JOB

    def step(self, current_schedule) -> int:
        eligible_ops = current_schedule.get_eligible_operations()
        operations_remaining_dict = {}
        for (job_id, op_id), _ in eligible_ops.items():
            # Calculate operations remaining for this job
            operations_remaining = len(current_schedule.instance[job_id]) - op_id
            operations_remaining_dict[job_id] = operations_remaining

        job_key = max(operations_remaining_dict, key=operations_remaining_dict.get)
        return job_key  # Return job_id


class LOR(Heuristic):
    """
    Least Operations Remaining (LOR) heuristic
    Prioritizes jobs with the least operations remaining
    """

    item_type = ItemDataType.JOB

    def step(self, current_schedule) -> int:
        eligible_ops = current_schedule.get_eligible_operations()
        operations_remaining_dict = {}
        for (job_id, op_id), _ in eligible_ops.items():
            # Calculate operations remaining for this job
            operations_remaining = len(current_schedule.instance[job_id]) - op_id
            operations_remaining_dict[job_id] = operations_remaining

        job_key = min(operations_remaining_dict, key=operations_remaining_dict.get)
        return job_key  # Return job_id


class RandomJob(Heuristic):
    """
    Random Choice heuristic
    Randomly selects an eligible job to schedule
    """

    item_type = ItemDataType.JOB

    def step(self, current_schedule) -> int:
        eligible_ops = current_schedule.get_eligible_operations()
        job_ids = list({job_id for (job_id, op_id) in eligible_ops.keys()})
        job_key = heuristic_rng.choice(job_ids)
        return job_key  # Return job_id


class FCFS(Heuristic):
    """
    First Come First Serve (FCFS) heuristic
    Processes jobs in order 0, 1, 2, ...
    """

    item_type = ItemDataType.JOB

    def step(self, current_schedule) -> int:
        eligible_ops = current_schedule.get_eligible_operations()
        if not eligible_ops:
            raise ValueError("No eligible operations")
        first_pair = next(iter(eligible_ops))
        return first_pair[0]


class EST(Heuristic):
    """
    Earliest Start Time (EST)
    Prioritizes the job that can start the soonest to minimize machine idle time.
    """

    item_type = ItemDataType.JOB

    def step(self, current_schedule) -> int:
        eligible_ops = current_schedule.get_eligible_operations()

        start_times = {}
        for job_id, op_id in eligible_ops:
            # Calculate when this specific op can effectively start
            start_times[job_id] = current_schedule.get_earliest_start_time(
                job_id, op_id
            )

        # Pick the job that starts earliest
        # Tie-breaker: If start times are equal, pick the one with smaller job_id
        best_job = min(start_times, key=lambda k: (start_times[k], k))
        return best_job


JOB_HEURISTICS_REGISTRY = {
    "spt": SPT,
    "smpt": SMPT,
    "lpt": LPT,
    "mwr": MWR,
    "lwr": LWR,
    "fddmwr": FDDMWR,
    "mor": MOR,
    "lor": LOR,
    "random": RandomJob,
    "fcfs": FCFS,
    "est": EST,
}


def job_heuristic_factory(name: str, random_nth_step: int | None = None) -> Heuristic:
    """
    Factory method to create job heuristic instances.
    Supports standard names (e.g. 'spt') and non-delay variants (e.g. 'nondelay_spt').
    """
    name = name.lower().strip()

    # Check for the wrapper prefix
    is_nondelay = name.startswith("nondelay_")

    # Extract the base name (e.g., 'nondelay_spt' -> 'spt')
    base_name = name.replace("nondelay_", "") if is_nondelay else name

    if base_name not in JOB_HEURISTICS_REGISTRY:
        raise ValueError(f"Unknown job heuristic: {base_name} (derived from '{name}')")

    # 1. Instantiate the base heuristic (e.g., SPT())
    heuristic_instance = JOB_HEURISTICS_REGISTRY[base_name]()

    # 2. Wrap it if requested
    if is_nondelay:
        heuristic_instance = NonDelay(heuristic_instance)

    if random_nth_step is not None and random_nth_step > 0:
        heuristic_instance = RandomStepHeuristic(heuristic_instance, random_nth_step)

    return heuristic_instance


def solve_jobinstance_with_heuristics(
    job_heuristic: Heuristic,
    schedule: Schedule,
) -> tuple[int, Heuristic]:
    done = schedule.is_complete()
    while not done:
        job_id = job_heuristic.step(schedule)
        schedule.schedule_job(job_id)
        done = schedule.is_complete()

    return (schedule.get_makespan(), job_heuristic)


if __name__ == "__main__":
    h = job_heuristic_factory("spt")
    print(h)
