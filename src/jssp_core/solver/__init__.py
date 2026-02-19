"""
JSSP Core Solvers

This module contains solver implementations and base classes for solving
Job Shop Scheduling Problems.

Classes:
    SolverType: Enumeration of solver types (heuristic, optimal, ML, etc.)
    JSSPSolverBase: Abstract base class for all JSSP solvers
    JSSPHeuristicSolver: Solver using priority/dispatching rules
    JSSPOptimalSolver: Optimal solver using constraint programming

Functions:
    compare_heuristics: Compare multiple heuristic approaches on an instance
"""

from jssp_core.solver.base import Heuristic, JSSPSolverBase, SolverType
from jssp_core.solver.comparison import compare_heuristics
from jssp_core.solver.heuristic_solver import JSSPHeuristicSolver
from jssp_core.solver.optimal import JSSPOptimalSolver


__all__ = [
    "SolverType",
    "JSSPSolverBase",
    "JSSPHeuristicSolver",
    "JSSPOptimalSolver",
    "compare_heuristics",
    "Heuristic",
]
