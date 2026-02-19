from __future__ import annotations

from collections.abc import Iterable

from jssp_core.instances import JSSPInstance
from jssp_core.schedule import Schedule
from jssp_core.solver.heuristics import (
    JOB_HEURISTICS_REGISTRY,
    job_heuristic_factory,
)


class JSSPHeuristicSolver:
    """
    Thin wrapper around the modern heuristic implementations to keep the legacy
    solver interface while delegating all decisions to the classes in
    heuristics.py.
    """

    def __init__(self, instance: JSSPInstance):
        self.instance = instance
        self.num_jobs = len(instance)
        self.num_machines = max((m for job in instance for m, _ in job), default=-1) + 1
        self.num_operations = sum(len(job) for job in instance)

    def _run(
        self,
        heuristic_name: str,
        partial_schedule: Schedule | None = None,
        steps: int | None = None,
    ) -> Schedule:
        heuristic = job_heuristic_factory(heuristic_name)
        schedule = (
            partial_schedule.copy() if partial_schedule else Schedule(self.instance)
        )

        remaining_steps = steps if steps is not None and steps > 0 else None
        while not schedule.is_complete() and (
            remaining_steps is None or remaining_steps > 0
        ):
            job_id = heuristic.step(schedule)
            schedule.schedule_job(int(job_id))
            if remaining_steps is not None:
                remaining_steps -= 1

        return schedule

    def solve_with_heuristic(self, heuristic: str) -> Schedule:
        """Solve an instance using a named heuristic."""
        return self._run(heuristic)

    def continue_from_partial_solution(
        self, partial_solution: Schedule, heuristic: str = "spt", steps: int = 0
    ) -> Schedule:
        """Resume scheduling from an existing partial schedule."""
        step_limit = None if steps == 0 else steps
        return self._run(heuristic, partial_schedule=partial_solution, steps=step_limit)

    def solve_from_env_state(self, env, heuristic: str = "spt") -> Schedule:
        """
        Rebuild a partial schedule from an environment and finish it with a heuristic.
        """
        partial_solution = Schedule(env.instance)
        for (job_id, _), _ in env.scheduled.items():
            partial_solution.schedule_job(job_id)
        return self.continue_from_partial_solution(partial_solution, heuristic)

    def solve_from_action_sequence(
        self, action_sequence: Iterable[int], heuristic: str = "spt"
    ) -> Schedule:
        """
        Apply a sequence of job actions and then finish the schedule with a heuristic.
        """
        partial_solution = Schedule(self.instance)
        for job_id in action_sequence:
            partial_solution.schedule_job(int(job_id))
        return self.continue_from_partial_solution(partial_solution, heuristic)

    def shortest_processing_time(self) -> Schedule:
        return self.solve_with_heuristic("spt")

    def longest_processing_time(self) -> Schedule:
        return self.solve_with_heuristic("lpt")

    def most_work_remaining(self) -> Schedule:
        return self.solve_with_heuristic("mwr")

    def least_work_remaining(self) -> Schedule:
        return self.solve_with_heuristic("lwr")

    def first_come_first_served(self) -> Schedule:
        return self.solve_with_heuristic("fcfs")

    def random_solution(self) -> Schedule:
        return self.solve_with_heuristic("random")

    def solve_all_heuristics(
        self, due_dates: list[int] | None = None
    ) -> dict[str, Schedule]:
        """Solve the instance with every available heuristic."""
        results: dict[str, Schedule] = {}
        for name in JOB_HEURISTICS_REGISTRY.keys():
            heuristic = job_heuristic_factory(name)
            results[name.upper()] = heuristic.solve(self.instance)
        return results
