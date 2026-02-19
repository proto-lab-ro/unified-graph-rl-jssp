import pytest

from jssp_core.models.operation import OperationInfo, ScheduledOperation
from jssp_core.schedule import Schedule
from jssp_core.solver.heuristic_solver import JSSPHeuristicSolver
from jssp_core.solver.heuristics import (
    JOB_HEURISTICS_REGISTRY,
    MWR,
    SMPT,
    SPT,
    job_heuristic_factory,
)


def test_operation_repr_helpers():
    op = OperationInfo(job_id=1, op_id=2, machine=3, duration=10)
    scheduled = ScheduledOperation(operation=op, start_time=5, end_time=15)

    assert repr(op) == "J1O2(M3,10)"
    assert repr(scheduled) == "J1O2(M3,10)[5-15]"


def test_job_heuristic_factory_is_case_insensitive():
    heuristic = job_heuristic_factory("SpT")
    assert isinstance(heuristic, SPT)


def test_job_heuristic_factory_unknown():
    with pytest.raises(ValueError, match="Unknown job heuristic"):
        job_heuristic_factory("does-not-exist")


def test_spt_picks_shortest_available_job():
    schedule = Schedule([[(0, 8)], [(1, 3)]])

    next_job = SPT().step(schedule)

    assert next_job == 1


def test_mwr_prefers_job_with_more_remaining_work():
    schedule = Schedule([[(0, 2), (1, 10)], [(1, 4)]])

    next_job = MWR().step(schedule)

    assert next_job == 0  # remaining work 12 vs 4


def test_smpt_prefers_earliest_ready_machine():
    schedule = Schedule([[(0, 2)], [(1, 2)]])
    schedule.machine_ready_time[0] = 5  # Machine 0 is busy longer than machine 1

    next_job = SMPT().step(schedule)

    assert next_job == 1


def test_solver_solves_all_heuristics():
    instance = [[(0, 3), (1, 2)], [(1, 4), (0, 1)]]
    solver = JSSPHeuristicSolver(instance)

    results = solver.solve_all_heuristics()

    assert set(results.keys()) == {name.upper() for name in JOB_HEURISTICS_REGISTRY}
    assert all(schedule.is_complete() for schedule in results.values())


def test_continue_from_partial_solution_preserves_progress():
    instance = [[(0, 2), (1, 2)], [(1, 3)]]
    partial = Schedule(instance)
    partial.schedule_job(0)
    solver = JSSPHeuristicSolver(instance)

    solved = solver.continue_from_partial_solution(partial, heuristic="lpt")

    assert solved.is_complete()
    assert (0, 0) in solved.scheduled
    assert solved.scheduled[(0, 0)] == partial.scheduled[(0, 0)]
