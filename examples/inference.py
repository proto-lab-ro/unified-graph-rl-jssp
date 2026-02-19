"""
Minimal example demonstrating how to load a pre-trained GNN model and solve a JSSP instance.
"""

from jssp_core.instances.generators import RandomInstanceGenerator
from jssp_gnn.solver import GnnMatrixSolver


def main():
    # 1. Load a pre-trained model package
    model_package = "examples/20x20_model.tar.gz"  # Update with your model path

    try:
        solver = GnnMatrixSolver.from_package(model_package, device="cpu")
    except FileNotFoundError:
        print(f"Error: Model package not found at {model_package}")
        return

    # 2. Create a JSSP instance (6 jobs, 6 machines)
    instance = RandomInstanceGenerator(num_jobs=6, num_machines=6).generate()

    # 3. Solve the instance with the GNN solver
    schedule = solver.solve(instance)

    # 4. Results
    print(f"Makespan: {schedule.get_makespan()}")
    print(f"Schedule Complete: {schedule.is_complete()}")


if __name__ == "__main__":
    main()
