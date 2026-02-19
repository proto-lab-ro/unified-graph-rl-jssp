"""
Tests for JSSP Optimal Constraint Programming Solver
"""

import os
import random
import sys
from collections.abc import Iterable
from unittest.mock import patch

import pytest


# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from jssp_core.instances import F3X3_INSTANCE, _parse_instance
from jssp_core.schedule import Schedule
from jssp_core.solver.heuristics import (
    JOB_HEURISTICS_REGISTRY,
    job_heuristic_factory,
)


# Import ORTOOLS_AVAILABLE for test skipping
try:
    from ortools.sat.python import cp_model  # noqa: F401

    ORTOOLS_AVAILABLE = True
except ImportError:
    ORTOOLS_AVAILABLE = False

# Only run tests if OR-Tools is available
pytestmark = pytest.mark.skipif(not ORTOOLS_AVAILABLE, reason="OR-Tools not available")


# Helper functions ------------------------------------------------------------
def _run_single_job_heuristic(instance, heuristic_name: str) -> Schedule:
    """Produce a schedule for a specific heuristic from heuristics."""
    heuristic = job_heuristic_factory(heuristic_name)
    if heuristic_name.lower() == "spt":
        random.seed(0)
    return heuristic.solve(instance)


def _solve_with_job_heuristics(
    instance, heuristic_names: Iterable[str] | None = None
) -> dict[str, Schedule]:
    """Run all requested heuristics on an instance using the new implementations."""
    names = heuristic_names or JOB_HEURISTICS_REGISTRY.keys()
    return {name.upper(): _run_single_job_heuristic(instance, name) for name in names}


@pytest.fixture
def small_instance():
    """Small 3x3 JSSP instance for testing"""
    return _parse_instance(F3X3_INSTANCE)


@pytest.fixture
def optimal_solver(small_instance):
    """Create optimal solver for small instance"""
    from jssp_core.solver.optimal import JSSPOptimalSolver

    solver = JSSPOptimalSolver(time_limit_seconds=30)
    solver.set_instance(small_instance)
    return solver


def test_optimal_solver_creation(optimal_solver):
    """Test that optimal solver can be created and initialized"""
    assert optimal_solver is not None
    assert optimal_solver.num_jobs == 3
    assert optimal_solver.num_machines == 3
    assert optimal_solver.num_operations == 9
    assert optimal_solver.time_limit_seconds == 30


def test_optimal_solver_solves_small_instance(optimal_solver):
    """Test that optimal solver can solve small instance"""
    solution, status = optimal_solver.solve_optimal(verbose=False)

    assert solution is not None
    assert solution.is_valid()
    assert solution.get_makespan() > 0

    # Check status
    from ortools.sat.python import cp_model

    assert status in (cp_model.OPTIMAL, cp_model.FEASIBLE)

    # Check that all operations are scheduled
    assert len(solution.scheduled) == optimal_solver.num_operations


def test_optimal_solver_solve_method(optimal_solver, small_instance):
    """Test the solve() method which returns just the schedule"""
    solution = optimal_solver.solve(small_instance)
    assert isinstance(solution, Schedule)
    assert solution.is_valid()


def test_optimal_solver_solve_raises_on_failure(optimal_solver, small_instance):
    """Test that solve() raises RuntimeError when no solution is found"""
    # Mock solve_optimal to return None
    with patch.object(optimal_solver, "solve_optimal", return_value=(None, 0)):
        with pytest.raises(RuntimeError, match="No solution found"):
            optimal_solver.solve(small_instance)


def test_optimal_solver_with_timeout(optimal_solver):
    """Test optimal solver with timeout method"""
    solution, is_optimal = optimal_solver.solve_with_timeout(
        timeout_seconds=30, verbose=False
    )

    assert solution is not None
    assert solution.is_valid()
    assert isinstance(is_optimal, bool)


def test_optimal_solver_vs_heuristics(small_instance):
    """Test that optimal solver finds better or equal solutions than heuristics"""
    from jssp_core.solver.optimal import JSSPOptimalSolver

    # Get best heuristic solution
    heuristic_results = _solve_with_job_heuristics(small_instance)
    best_heuristic_makespan = min(
        sol.get_makespan() for sol in heuristic_results.values()
    )

    # Get optimal solution
    optimal_solver = JSSPOptimalSolver(time_limit_seconds=30)
    optimal_solution, _ = optimal_solver.solve_optimal(
        instance=small_instance, verbose=False
    )

    assert optimal_solution is not None
    optimal_makespan = optimal_solution.get_makespan()

    # Optimal should be better than or equal to best heuristic
    assert optimal_makespan <= best_heuristic_makespan


def test_compare_heuristics_with_optimal(small_instance):
    """Test enhanced compare_heuristics function with optimal solver"""
    from jssp_core.solver.optimal import JSSPOptimalSolver

    heuristic_schedules = _solve_with_job_heuristics(small_instance)

    optimal_solver = JSSPOptimalSolver(time_limit_seconds=30)
    optimal_schedule, _ = optimal_solver.solve_optimal(
        instance=small_instance, verbose=False
    )
    assert optimal_schedule is not None

    results = {
        name: {
            "valid": schedule.is_valid(),
            "makespan": schedule.get_makespan(),
            "solution": schedule,
        }
        for name, schedule in heuristic_schedules.items()
    }
    results["Optimal"] = {
        "valid": optimal_schedule.is_valid(),
        "makespan": optimal_schedule.get_makespan(),
        "solution": optimal_schedule,
    }

    # Check that optimal has the best makespan
    optimal_makespan = results["Optimal"]["makespan"]
    for name, data in results.items():
        if name != "Optimal":
            assert data["makespan"] >= optimal_makespan


def test_solve_with_info(optimal_solver, small_instance):
    """Test solve_with_info method"""
    from jssp_core.solver.base import SolveOutput

    result = optimal_solver.solve_with_info(small_instance)

    assert isinstance(result, SolveOutput)
    assert result.solution is not None
    assert result.solution.get_makespan() > 0
    assert "cp_model_status" in result.info


def test_get_config_hash(optimal_solver):
    """Test get_config_hash method"""
    hash_val = optimal_solver.get_config_hash()
    assert isinstance(hash_val, str)
    assert len(hash_val) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
