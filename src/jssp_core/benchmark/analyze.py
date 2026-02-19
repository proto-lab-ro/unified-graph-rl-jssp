import json
import os
from typing import Any

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch


def execute_query_as_df(
    conn: Any, query: str, params: list | tuple | None = None
) -> pd.DataFrame:
    """Execute query and return result as DataFrame, handling different connection types."""
    is_duckdb = isinstance(conn, duckdb.DuckDBPyConnection)
    if is_duckdb:
        if params:
            return conn.execute(query, params).df()
        return conn.execute(query).df()
    else:
        # Assume pyodbc or generic DBAPI connection compatible with pandas
        return pd.read_sql(query, conn, params=params)


def get_all_results(conn: Any) -> pd.DataFrame:
    """Get all results as a DataFrame."""
    return execute_query_as_df(
        conn,
        """
        SELECT *
        FROM benchmark_records_normalized
        WHERE error IS NULL
        """,
    )


def get_runs_per_solver(
    conn: Any = None, data: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Get number of completed runs per solver."""
    if data is None:
        if conn is None:
            raise ValueError("Either conn or data must be provided")
        return execute_query_as_df(
            conn,
            """
            SELECT solver_id, COUNT(*) AS completed_runs
            FROM benchmark_records_normalized
            WHERE error IS NULL
            GROUP BY solver_id
            ORDER BY completed_runs DESC
            """,
        )
    else:
        return (
            data.groupby("solver_id")
            .size()
            .reset_index(name="completed_runs")
            .sort_values("completed_runs", ascending=False)
        )


def calculate_gap_to_literature_best(
    data: pd.DataFrame, literature_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Calculate gap to literature best solutions.

    Args:
        data (pd.DataFrame): Benchmark results DataFrame.
        literature_df (pd.DataFrame): DataFrame with columns 'instance_id' and 'best_makespan'.

    Returns:
        pd.DataFrame: DataFrame with an additional 'gap_to_literature_best' column.
    """
    df = data.copy()
    merged = df.merge(
        literature_df,
        on="instance_id",
        how="left",
        suffixes=("", "_literature_best"),
    )

    merged["gap_literature_best"] = np.where(
        merged["makespan_literature_best"] == 0,
        np.nan,
        (merged["makespan"] - merged["makespan_literature_best"])
        / merged["makespan_literature_best"],
    )
    return merged


def calculate_gap_to_baseline(
    data: pd.DataFrame,
    baseline_solver: str | None = None,
) -> pd.DataFrame:
    """
    Calculates gap to a baseline solver.
    If baseline solver is None the gap to best known solution in dataset is calculated.
    """
    df = data.copy()
    if baseline_solver:
        baseline_df = df[df["solver_id"] == baseline_solver][
            ["instance_id", "makespan"]
        ].rename(columns={"makespan": "baseline_makespan"})
        merged = df.merge(baseline_df, on="instance_id", how="left")
    else:
        best_df = (
            df.groupby("instance_id")["makespan"]
            .min()
            .reset_index()
            .rename(columns={"makespan": "baseline_makespan"})
        )
        merged = df.merge(best_df, on="instance_id", how="left")

    merged["gap"] = np.where(
        merged["baseline_makespan"] == 0,
        np.nan,
        (merged["makespan"] - merged["baseline_makespan"])
        / merged["baseline_makespan"],
    )
    return merged


def calculate_solver_stats(
    data: pd.DataFrame | None = None,
    baseline_solver: str | None = None,
) -> pd.DataFrame:
    """Calculate performance statistics by solver."""
    df = calculate_gap_to_baseline(data, baseline_solver)
    if df.empty:
        return pd.DataFrame()

    return (
        df.groupby(["solver_id", "solver_type"])
        .agg(
            mean_gap=("gap", "mean"),
            median_gap=("gap", "median"),
            mean_time=("computation_time_seconds", "mean"),
            mean_makespan=("makespan", "mean"),
            best_count=("gap", lambda x: (x <= 1e-6).sum()),
            runs=("run_id", "count"),
        )
        .sort_values("mean_gap")
    )


def create_solver_selector(
    data: pd.DataFrame,
    baseline_solver: str | None = None,
) -> "SolverSelector":
    """
    Start a solver selection chain.

    Returns:
        SolverSelector: A selector object initialized with current solver stats.
    """
    stats = calculate_solver_stats(
        data=data, baseline_solver=baseline_solver
    ).reset_index()
    return SolverSelector(stats)


def print_solver_performance_summary(
    data: pd.DataFrame,
    baseline_solver: str | None = None,
):
    """Print summary of solver performance"""
    stats = calculate_solver_stats(data=data, baseline_solver=baseline_solver)

    print("\n" + "=" * 80)
    print("SOLVER PERFORMANCE ANALYSIS")
    print("=" * 80)
    print(stats)


def calculate_performance_by_size(
    data: pd.DataFrame,
    baseline_solver: str | None = None,
) -> pd.DataFrame:
    """Calculate mean gap and makespan for each solver by problem size."""
    df = calculate_gap_to_baseline(data, baseline_solver=baseline_solver)
    if df.empty:
        return pd.DataFrame()

    # Create size column
    df["size"] = df["num_jobs"].astype(str) + "x" + df["num_machines"].astype(str)

    return (
        df.groupby(["size", "solver_id"])
        .agg(
            mean_gap=("gap", "mean"),
            mean_makespan=("makespan", "mean"),
            mean_time=("computation_time_seconds", "mean"),
            count=("run_id", "count"),
            num_jobs=("num_jobs", "first"),
            num_machines=("num_machines", "first"),
            solver_type=("solver_type", "first"),
        )
        .reset_index()
        .sort_values(["num_jobs", "num_machines"])
    )


def print_gap_by_size(data: pd.DataFrame, baseline_solver: str | None = None):
    """Prints a table with solvers as columns and problem sizes as rows."""
    df = calculate_performance_by_size(data, baseline_solver=baseline_solver)

    if df.empty:
        print("No results to display.")
        return

    pivot_df = df.pivot(index="size", columns="solver_id", values="mean_gap")

    print("\n" + "=" * 80)
    print("PERFORMANCE BY PROBLEM SIZE (Mean Gap)")
    print("=" * 80)
    print(pivot_df)
    return pivot_df


def plot_performance_scaling(
    conn: Any = None,
    metric: str = "mean_gap",
    output_path: str | None = None,
    solvers: list[str] | None = None,
    data: pd.DataFrame | None = None,
):
    """
    Plot performance scaling with problem size.

    Args:
        metric: 'mean_gap' or 'mean_time'
        output_path: Path to save the plot
        solvers: Optional list of solver IDs to include.
        data: Optional DataFrame to use instead of fetching from DB.
    """
    if data is None:
        if conn is None:
            raise ValueError("Either conn or data must be provided")
        data = get_all_results(conn)

    df = calculate_performance_by_size(data)
    if df.empty:
        print("No data to plot.")
        return

    if solvers:
        df = df[df["solver_id"].isin(solvers)]
        if df.empty:
            print("No data for selected solvers.")
            return

    plt.figure(figsize=(12, 6))

    # Ensure correct sorting for the x-axis
    df["size_rank"] = df["num_jobs"] * 1000 + df["num_machines"]
    df = df.sort_values("size_rank")

    sns.lineplot(
        data=df,
        x="size",
        y=metric,
        hue="solver_id",
        style="solver_type",
        markers=True,
        dashes=False,
    )

    metric_label = (
        "Mean Gap (%)" if metric == "mean_gap" else "Mean Computation Time (s)"
    )
    plt.title(f"Solver Scaling: {metric_label} vs Problem Size")
    plt.ylabel(metric_label)
    plt.xlabel("Problem Size (Jobs x Machines)")
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path)
        print(f"Plot saved to {output_path}")
    else:
        plt.show()


def plot_pareto_frontier(
    conn: Any = None,
    output_path: str | None = None,
    solvers: list[str] | None = None,
    data: pd.DataFrame | None = None,
):
    """
    Plot Mean Time vs Mean Gap to show the trade-off between speed and quality.
    """
    stats = calculate_solver_stats(conn, data=data).reset_index()

    if stats.empty:
        print("No data to plot.")
        return

    if solvers:
        stats = stats[stats["solver_id"].isin(solvers)]
        if stats.empty:
            print("No data for selected solvers.")
            return

    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        data=stats,
        x="mean_time",
        y="mean_gap",
        hue="solver_type",
        style="solver_id",
        s=150,
    )

    # Add labels
    # for i, row in stats.iterrows():
    #     plt.text(
    #         row["mean_time"],
    #         row["mean_gap"],
    #         row["solver_id"],
    #         fontsize=9,
    #         ha="left",
    #         va="bottom",
    #         xytext=(5, 5),
    #         textcoords="offset points",
    #     )

    plt.title("Solver Efficiency: Time vs Quality Trade-off")
    plt.xlabel("Mean Computation Time (s)")
    plt.ylabel("Mean Gap to Best (%)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path)
        print(f"Plot saved to {output_path}")
    else:
        plt.show()


def plot_win_rate(
    data: pd.DataFrame,
    baseline_solver: str | None = None,
    output_path: str | None = None,
    solvers: list[str] | None = None,
):
    """
    Plot the percentage of instances where each solver found the best solution.
    """
    df = calculate_gap_to_baseline(
        data=data, baseline_solver=baseline_solver
    )  # Gap to best known
    if df.empty:
        return

    if solvers:
        df = df[df["solver_id"].isin(solvers)]
        if df.empty:
            print("No data for selected solvers.")
            return

    # Best solution has gap 0 (or very close to 0 due to float precision)
    df["is_best"] = df["gap"] < 1e-6

    win_rates = df.groupby(["solver_id", "solver_type"])["is_best"].mean().reset_index()
    win_rates["win_rate_pct"] = win_rates["is_best"] * 100
    win_rates = win_rates.sort_values("win_rate_pct", ascending=False)

    plt.figure(figsize=(12, 6))
    sns.barplot(
        data=win_rates,
        x="solver_id",
        y="win_rate_pct",
        hue="solver_type",
        dodge=False,
    )

    plt.title("Solver Win Rate (% of instances with best known solution)")
    plt.ylabel("Win Rate (%)")
    plt.xlabel("Solver")
    plt.xticks(rotation=45)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path)
        print(f"Plot saved to {output_path}")
    else:
        plt.show()


def plot_boxplot(
    data: pd.DataFrame,
    baseline_solver: str | None = None,
    output_path: str | None = None,
    solvers: list[str] | None = None,
    return_plt: bool = False,
):
    """Generate a boxplot of gaps"""
    df = calculate_gap_to_baseline(data=data, baseline_solver=baseline_solver)
    if df.empty:
        print("No data to plot.")
        return

    if solvers:
        df = df[df["solver_id"].isin(solvers)]
        if df.empty:
            print("No data for selected solvers.")
            return

    plt.figure(figsize=(12, 6))
    # Sort solvers by mean gap
    solver_order = df.groupby("solver_id")["gap"].mean().sort_values().index
    sns.boxplot(data=df, x="solver_id", y="gap", order=solver_order)
    if not baseline_solver:
        baseline_solver = "Best Known"
    plt.title(f"Solver Performance Comparison (Gap to: {baseline_solver})")
    plt.ylabel("Gap (%)")
    plt.xlabel("Solver")
    plt.xticks(rotation=45)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path)
        print(f"Plot saved to {output_path}")
    elif return_plt:
        return plt
    else:
        plt.show()


def get_solver_comparison_data(
    data: pd.DataFrame,
    solvers: list[str],
    baseline_solver: str | None = None,
) -> pd.DataFrame:
    """Get makespan comparison data for specified solvers."""
    df = calculate_gap_to_baseline(data=data, baseline_solver=baseline_solver)
    # Filter for selected solvers
    df_filtered = df[df["solver_id"].isin(solvers)]

    if df_filtered.empty:
        return pd.DataFrame()

    # Pivot to get makespans side by side for each instance
    pivot_df = df_filtered.pivot_table(
        index="instance_id",
        columns="solver_id",
        values="makespan",
        aggfunc="mean",
    ).dropna()

    return pivot_df


def plot_solver_comparison(
    data: pd.DataFrame,
    solvers: list[str],
    baseline_solver: str | None = None,
    output_path: str | None = None,
):
    """Plot makespan comparison between two or more solvers."""
    pivot_df = get_solver_comparison_data(
        data=data, solvers=solvers, baseline_solver=baseline_solver
    )

    if pivot_df.empty:
        print(f"No common data found for the specified solvers: {solvers}")
        return

    available_solvers = [s for s in solvers if s in pivot_df.columns]
    if len(available_solvers) < 2:
        print(f"Need at least two valid solvers to compare. Found: {available_solvers}")
        return

    if len(available_solvers) == 2:
        solver1, solver2 = available_solvers

        plt.figure(figsize=(10, 10))
        sns.scatterplot(data=pivot_df, x=solver1, y=solver2)

        # Add diagonal line for reference (y=x)
        min_val = min(pivot_df[solver1].min(), pivot_df[solver2].min())
        max_val = max(pivot_df[solver1].max(), pivot_df[solver2].max())

        # Add some padding to the range
        padding = (max_val - min_val) * 0.05
        limit_min = min_val - padding
        limit_max = max_val + padding

        plt.plot(
            [limit_min, limit_max],
            [limit_min, limit_max],
            "r--",
            label="Equal Performance",
            alpha=0.7,
        )

        plt.title(f"Makespan Comparison: {solver1} vs {solver2}")
        plt.xlabel(f"{solver1} Makespan")
        plt.ylabel(f"{solver2} Makespan")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.axis("equal")

    else:
        # Pairplot for more than 2 solvers
        g = sns.pairplot(pivot_df[available_solvers], diag_kind="kde")
        g.fig.suptitle("Solver Makespan Comparison Matrix", y=1.02)

        # Add diagonal lines to off-diagonal plots
        for i, row_var in enumerate(available_solvers):
            for j, col_var in enumerate(available_solvers):
                if i != j:
                    ax = g.axes[i, j]
                    min_val = min(pivot_df[row_var].min(), pivot_df[col_var].min())
                    max_val = max(pivot_df[row_var].max(), pivot_df[col_var].max())
                    ax.plot([min_val, max_val], [min_val, max_val], "r--", alpha=0.5)

    if output_path:
        plt.savefig(output_path)
        print(f"Plot saved to {output_path}")
    else:
        plt.show()


def get_solver_logits(
    data: pd.DataFrame,
    max_instances: int = 10,
) -> dict[str, list[tuple[str, np.ndarray]]]:
    """
    Extract logits for each solver.

    Returns:
        dict: Mapping from solver_id to list of (instance_id, logits) tuples.
    """

    df = data

    solver_logits = {}
    count = 0

    for _, row in df.iterrows():
        if count >= max_instances:
            break

        metrics = row["additional_metrics"]
        # metrics might be a dict (if DuckDB converted it) or a string (JSON)
        if isinstance(metrics, str):
            try:
                metrics = json.loads(metrics)
            except Exception:
                continue

        if not isinstance(metrics, dict):
            continue

        logits = metrics.get("logits")
        if logits is not None:
            # Convert logits to numpy array
            if isinstance(logits, list):
                logits = np.array(logits)
            elif isinstance(logits, torch.Tensor):
                logits = logits.cpu().numpy()

            solver_id = row["solver_id"]
            instance_id = row["instance_id"]

            if solver_id not in solver_logits:
                solver_logits[solver_id] = []

            solver_logits[solver_id].append((instance_id, logits))
            count += 1

    return solver_logits


def plot_logits_heatmap(
    data: pd.DataFrame,
    output_dir: str | None = None,
):
    """Generate heatmaps of logits for each solver."""
    solver_logits = get_solver_logits(data=data)

    for solver_name, runs in solver_logits.items():
        all_logits = []
        run_indices = []
        current_idx = 0

        for instance_idx, logits in runs:
            all_logits.append(logits)
            run_indices.append(
                (current_idx, current_idx + logits.shape[0], instance_idx)
            )
            current_idx += logits.shape[0]

        if all_logits:
            combined_logits = np.vstack(all_logits)

            plt.figure(figsize=(12, 8))
            plt.imshow(combined_logits, cmap="viridis", aspect="auto")
            plt.colorbar(label="Logit Value")

            # Add horizontal lines to separate runs
            for start, end, instance_idx in run_indices:
                plt.axhline(y=end - 0.5, color="white", linestyle="--", linewidth=0.5)
                # Add text labels on the right side
                plt.text(
                    combined_logits.shape[1] + 0.5,
                    (start + end) / 2,
                    f"Inst {instance_idx}",
                    va="center",
                    fontsize=8,
                )

            plt.title(f"Combined Logits Heatmap - {solver_name}")
            plt.xlabel("Actions")
            plt.ylabel("Steps (concatenated)")
            plt.tight_layout()

            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                output_path = os.path.join(
                    output_dir, f"logits_heatmap_{solver_name}.png"
                )
                plt.savefig(output_path)
                print(f"Plot saved to {output_path}")
                plt.close()
            else:
                plt.show()
