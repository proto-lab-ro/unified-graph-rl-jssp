import argparse
import pickle
from datetime import datetime
from pathlib import Path

from jssp_core.instances.generators import initialize_instance_generator
from jssp_core.reproducibility import set_seed


def generate_benchmark_set(
    n_instances: int,
    sizes: list[tuple[int, int]],
    output_dir: str,
    seed: int = 42,
    filename: str | None = None,
    prefix: str = "random",
):
    """
    Generates a set of benchmark instances and saves them to a pickle file.

    Args:
        n_instances: Number of instances per size.
        sizes: List of (n_jobs, n_machines) tuples.
        output_dir: Directory to save the file.
        seed: Random seed.
        filename: Optional filename. If None, one is generated.
    """
    set_seed(seed)

    generators = []
    for n_jobs, n_machines in sizes:
        print(f"Generating {n_instances} instances of size {n_jobs}x{n_machines}...")
        gen = initialize_instance_generator(
            "random_uniform",
            num_jobs=n_jobs,
            num_machines=n_machines,
            stop_count=n_instances,
            generator_kwargs={
                "min_duration": 1,
                "max_duration": 99,
            },
        )
        generators.append(gen)

    benchmark_instances = []
    for gen in generators:
        for idx, instance in enumerate(gen, start=1):
            instance.name = f"{prefix}_{instance.num_jobs()}x{instance.num_machines()}_{idx}"  # Set the name attribute
            benchmark_instances.append(instance)

    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        size_str = "-".join([f"{j}x{m}" for j, m in sizes])
        # Truncate size string if too long
        if len(size_str) > 20:
            size_str = f"{len(sizes)}_sizes"
        filename = f"{timestamp}_{n_instances}_instances_{size_str}.pkl"

    output_path = Path(output_dir) / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Saving {len(benchmark_instances)} instances to {output_path}...")
    with open(output_path, "wb") as f:
        pickle.dump(benchmark_instances, f)

    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate JSSP benchmark instances.")
    parser.add_argument(
        "--n_instances", type=int, default=100, help="Number of instances per size."
    )
    parser.add_argument(
        "--output_dir", type=str, default="benchmark_results", help="Output directory."
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--filename", type=str, default=None, help="Output filename.")

    # Default sizes from the user's snippet
    # [(6, 6), (10, 10), (15, 15), (20, 20), (30, 20)]
    # We can allow passing sizes as a string "6x6,10x10"
    parser.add_argument(
        "--sizes",
        type=str,
        default="6x6,10x10,15x15,20x20,30x20",
        help="Comma-separated list of sizes (e.g., '6x6,10x10').",
    )

    parser.add_argument(
        "--prefix",
        type=str,
        default="random",
        help="Prefix used for generated instance names.",
    )

    args = parser.parse_args()

    # Parse sizes
    sizes_list = []
    for s in args.sizes.split(","):
        j, m = map(int, s.split("x"))
        sizes_list.append((j, m))

    generate_benchmark_set(
        n_instances=args.n_instances,
        sizes=sizes_list,
        output_dir=args.output_dir,
        seed=args.seed,
        filename=args.filename,
        prefix=args.prefix,
    )
