"""
Unit tests for the CP-SAT optimal solver. These exercise the real OR-Tools
implementation when available and verify the ImportError guard otherwise.
"""

import pytest

from jssp_core.instances import (
    FT06_INSTANCE,
    _load_instance,
    get_instance,
)
from jssp_core.solver.optimal import ORTOOLS_AVAILABLE, JSSPOptimalSolver


@pytest.mark.unit
@pytest.mark.skipif(ORTOOLS_AVAILABLE, reason="OR-Tools installed; guard path not hit")
def test_optimal_solver_import_guard_when_missing():
    """Without OR-Tools installed, construction should fail fast."""
    with pytest.raises(ImportError):
        JSSPOptimalSolver(time_limit_seconds=10)


@pytest.mark.unit
@pytest.mark.skipif(not ORTOOLS_AVAILABLE, reason="OR-Tools not installed")
def test_optimal_solver_solves_small_instance():
    """Solve the 3x3 toy instance optimally within a short timeout."""
    solver = JSSPOptimalSolver()
    instance = get_instance("f3x3")

    schedule = solver.solve_optimal(
        instance=instance, time_limit_seconds=10, verbose=False
    )[0]

    assert schedule is not None
    assert schedule.is_complete()
    assert schedule.is_valid()
    assert schedule.get_makespan() > 0
    assert len(schedule.scheduled) == solver.num_operations


@pytest.mark.unit
@pytest.mark.skipif(not ORTOOLS_AVAILABLE, reason="OR-Tools not installed")
def test_optimal_solver_timeout_wrapper():
    """Timeout wrapper should surface a solution and optimistic optimal flag."""
    instance = get_instance("f3x3")
    solver = JSSPOptimalSolver()
    solver.set_instance(instance)

    schedule, is_optimal = solver.solve_with_timeout(timeout_seconds=10, verbose=False)

    assert schedule is not None
    assert schedule.is_complete()
    assert schedule.is_valid()
    assert isinstance(is_optimal, bool)


@pytest.mark.unit
@pytest.mark.skipif(not ORTOOLS_AVAILABLE, reason="OR-Tools not installed")
def test_optimal_solver_ft06_makespan():
    """FT06 instance lower bound is 55; solution should respect it."""
    instance = get_instance(FT06_INSTANCE)
    solver = JSSPOptimalSolver()

    schedule = solver.solve_optimal(
        instance=instance, time_limit_seconds=30, verbose=False
    )[0]

    assert schedule is not None
    assert schedule.is_complete()
    assert schedule.is_valid()
    assert schedule.get_makespan() == 55


@pytest.mark.unit
@pytest.mark.skipif(not ORTOOLS_AVAILABLE, reason="OR-Tools not installed")
def test_optimal_solver_ft10_lower_bound():
    """FT10 optimal makespan should match the known 930 optimum."""
    instance = _load_instance("jssp_instances/ft10")
    solver = JSSPOptimalSolver()

    schedule = solver.solve_optimal(
        instance=instance, time_limit_seconds=60, verbose=False
    )[0]

    assert schedule is not None
    assert schedule.is_complete()
    assert schedule.is_valid()
    assert schedule.get_makespan() == 930


@pytest.mark.unit
@pytest.mark.skipif(not ORTOOLS_AVAILABLE, reason="OR-Tools not installed")
def test_optimal_solver_la01_optimal():
    """LA01 optimal makespan is 666."""
    instance = _load_instance("jssp_instances/la01")
    solver = JSSPOptimalSolver()

    schedule = solver.solve_optimal(
        instance=instance, time_limit_seconds=60, verbose=False
    )[0]

    assert schedule is not None
    assert schedule.is_complete()
    assert schedule.is_valid()
    assert schedule.get_makespan() == 666
