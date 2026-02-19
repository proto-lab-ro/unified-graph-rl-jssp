"""
JSSP Core Package

This package contains the core JSSP functionality independent of any ML/RL framework:
- Schedule: Core scheduling logic
- Environment: Gymnasium interface for JSSP
- Heuristics: Various heuristic algorithms and optimal constraint programming solver
- Instance handling: Problem instance loading and generation
- Utilities: Common utilities and integration functions
"""

from jssp_core.environments.jssp import JSSPEnv
from jssp_core.instances import get_instance
from jssp_core.reproducibility import set_seed
from jssp_core.schedule import Schedule
from jssp_core.solver.heuristic_solver import JSSPHeuristicSolver


# Optional optimal solver (requires OR-Tools)
try:
    from jssp_core.solver.optimal import JSSPOptimalSolver

    OPTIMAL_SOLVER_AVAILABLE = True
except ImportError:
    JSSPOptimalSolver = None  # type: ignore
    OPTIMAL_SOLVER_AVAILABLE = False


__version__ = "1.0.0"

__all__ = [
    "Schedule",
    "JSSPEnv",
    "JSSPHeuristicSolver",
    "set_seed",
    "get_instance",
]

# Add optimal solver to exports if available
if OPTIMAL_SOLVER_AVAILABLE:
    __all__.append("JSSPOptimalSolver")
