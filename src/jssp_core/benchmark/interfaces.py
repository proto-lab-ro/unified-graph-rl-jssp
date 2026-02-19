from typing import Protocol

from jssp_core.instances.jssp import JSSPInstance
from jssp_core.solver.base import SolveOutput, SolverType


class Solver(Protocol):
    """Protocol for any solver that can be benchmarked"""

    @property
    def name(self) -> str:
        """Name of the solver"""
        ...

    def get_type(self) -> SolverType:
        """Type of the solver"""
        ...

    def solve_with_info(self, instance: JSSPInstance) -> SolveOutput:
        """
        Solve the given instance and return the result.

        Args:
            instance: The JSSP instance to solve

        Returns:
            SolveOutput containing solution and info.
        """
        ...
