import importlib.util

from jssp_core.solver.heuristic_solver import JSSPHeuristicSolver
from jssp_core.solver.optimal import JSSPOptimalSolver


ORTOOLS_AVAILABLE = importlib.util.find_spec("ortools") is not None


def compare_heuristics(
    instance,
    due_dates: list[int] | None = None,
    verbose: bool = True,
    include_optimal: bool = False,
    optimal_time_limit: int = 300,
):
    """
    Compare all heuristics on a given instance and return results

    Args:
        instance: JSSP instance
        due_dates: Due dates for jobs (optional)
        verbose: Whether to print results
        include_optimal: Whether to include optimal CP solver
        optimal_time_limit: Time limit for optimal solver in seconds
    """
    solver = JSSPHeuristicSolver(instance)

    print(
        f"Solving JSSP instance with {solver.num_jobs} jobs, "
        f"{solver.num_machines} machines, {solver.num_operations} operations"
    )
    print("=" * 80)

    results = solver.solve_all_heuristics(due_dates)

    # Add optimal solution if requested and OR-Tools is available
    if include_optimal:
        if ORTOOLS_AVAILABLE:
            try:
                optimal_solver = JSSPOptimalSolver(instance)
                print("\nSolving optimally with constraint programming...")
                optimal_solution = optimal_solver.solve_optimal(
                    optimal_time_limit, verbose=verbose
                )
                if optimal_solution:
                    results["Optimal"] = optimal_solution
                else:
                    print("Could not find optimal solution within time limit")
            except Exception as e:
                print(f"Error solving optimally: {e}")
        else:
            print("OR-Tools not available - skipping optimal solver")
            print("Install with: pip install ortools")

    # Sort by makespan
    sorted_results = sorted(results.items(), key=lambda x: x[1].get_makespan())

    if verbose:
        print(f"\n{'Method':<12} {'Makespan':<10} {'Valid':<8} {'Gap%':<8}")
        print("-" * 45)

    performance_data = {}
    best_makespan = sorted_results[0][1].get_makespan()

    for name, solution in sorted_results:
        makespan = solution.get_makespan()
        is_valid = solution.is_valid()
        gap = (
            ((makespan - best_makespan) / best_makespan * 100)
            if best_makespan > 0
            else 0
        )

        performance_data[name] = {
            "makespan": makespan,
            "valid": is_valid,
            "solution": solution,
            "gap_percent": gap,
        }

        if verbose:
            print(f"{name:<12} {makespan:<10} {is_valid:<8} {gap:<8.1f}")

    if verbose:
        print("=" * 80)
        best_name, best_solution = sorted_results[0]
        print(
            f"Best solution: {best_name} with makespan {best_solution.get_makespan()}"
        )

        if "Optimal" in results:
            optimal_makespan = results["Optimal"].get_makespan()
            print(f"Optimal makespan: {optimal_makespan}")

            # Show gaps from optimal for all heuristics
            print("\nGaps from optimal:")
            for name, data in performance_data.items():
                if name != "Optimal":
                    gap = (data["makespan"] - optimal_makespan) / optimal_makespan * 100
                    print(f"  {name}: {gap:.1f}%")

    return performance_data
