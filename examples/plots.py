import re
from pathlib import Path

import duckdb
import matplotlib.collections as mcoll
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter

from jssp_core.benchmark import analyze


# --- Configuration ---
TARGET_ML_SOLVER = (
    "260124_10-12-18-20x20_reward_mixer_0-20x20-6941b4af4dbc__policy_module_final"
)
HEURISTIC_IDS = ["random", "mwr", "mor", "fddmwr"]
BENCHMARKS = ["jh", "ta", "la", "sm", "bm"]
SOLVERS_TO_PLOT = HEURISTIC_IDS + [TARGET_ML_SOLVER]

LABEL_MAP = {
    "random": "Random",
    "mor": "MOR",
    "mwr": "MWKR",
    "fddmwr": "FDD/MWR",
    TARGET_ML_SOLVER: "Ours (Single)",
}
COLOR_MAP = {
    "Random": "#FFFFFF",
    "MOR": "#DFDFDF",
    "MWKR": "#BFBFBF",
    "FDDMWR": "#9F9F9F",
    "Ours (Single)": "#7A7A7A",
}


def configure_plotting(style="ieee"):
    """Sets up the plotting style."""
    sns.set_style("whitegrid")
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42

    if style == "ieee":
        plt.rcParams.update(
            {
                "font.family": "serif",
                "font.size": 8,
                "axes.labelsize": 9,
                "axes.titlesize": 9,
                "xtick.labelsize": 8,
                "ytick.labelsize": 8,
                "legend.fontsize": 7,
                "lines.linewidth": 1,
                "grid.color": "#e0e0e0",
                "grid.linestyle": "--",
            }
        )
    else:
        plt.rcParams.update(
            {
                "font.family": "serif",
                "font.size": 11,
                "axes.labelsize": 12,
                "axes.titlesize": 12,
                "xtick.labelsize": 10,
                "ytick.labelsize": 10,
            }
        )


def load_data():
    """Loads benchmark results and calculates gaps."""
    print("Loading data...")
    conn = duckdb.connect(":memory:")

    # Load raw results
    df_raw = conn.execute("SELECT * FROM 'examples/benchmark_results.parquet'").df()

    # Load literature best solutions
    conn.execute("""
        CREATE OR REPLACE TABLE literature_solutions AS
        SELECT *, name as instance_id
        FROM read_csv_auto('jssp_instances/solutions/best_benchmark_solutions.csv');
    """)
    df_literature_best = conn.execute("SELECT * FROM literature_solutions").df()

    # Calculate gaps
    df = analyze.calculate_gap_to_literature_best(
        data=df_raw, literature_df=df_literature_best
    )
    df = analyze.calculate_gap_to_baseline(data=df, baseline_solver="optimal")

    # Add helper columns
    df["solution_list"] = df.solution.apply(
        lambda x: [
            int(i)
            for i in str(x)
            .replace(" ", "")
            .replace("[", "")
            .replace("]", "")
            .split(",")
            if i
        ]
    )

    print(f"Data loaded. Rows: {len(df)}")
    return df, conn


def plot_performance_boxplot(df):
    """Generates the overall solver performance boxplot."""
    print("Generating performance boxplot...")
    tmp_df = df[df["benchmark_name"].isin(BENCHMARKS)]
    plot_data = [
        tmp_df.loc[tmp_df["solver_id_norm"] == sid, "gap"].values
        for sid in SOLVERS_TO_PLOT
    ]
    labels = [LABEL_MAP.get(s, s) for s in SOLVERS_TO_PLOT]
    colors = [COLOR_MAP.get(label, "#FFFFFF") for label in labels]

    fig, ax = plt.subplots(figsize=(3.5, 2.3))

    box = ax.boxplot(
        plot_data,
        patch_artist=True,
        widths=0.6,
        showmeans=True,
        showfliers=True,
        medianprops={"color": "black", "linewidth": 1},
        meanprops={
            "marker": "^",
            "markerfacecolor": "white",
            "markeredgecolor": "black",
            "markersize": 4,
        },
        flierprops={"marker": ".", "color": "black", "markersize": 2, "alpha": 0.5},
        whiskerprops={"color": "black", "linewidth": 0.8},
        capprops={"color": "black", "linewidth": 0.8},
    )

    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_edgecolor("black")
        patch.set_linewidth(0.8)

    legend_handles = [
        Patch(facecolor=COLOR_MAP.get(label, "#fff"), edgecolor="black", label=label)
        for label in labels
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper right",
        frameon=False,
        ncol=1,
        handletextpad=0.5,
        borderaxespad=0.2,
    )

    ax.set_xticks([])
    ax.yaxis.set_major_formatter(
        FuncFormatter(lambda x, pos: f"{x:.2g}" if x != 0 else "0")
    )
    ax.set_ylabel("Optimality Gap (%)")
    ax.yaxis.grid(True, linestyle=":", alpha=0.5)
    sns.despine(bottom=True)

    plt.tight_layout(pad=0.2)
    plt.savefig("plot_performance.pdf", format="pdf", bbox_inches="tight", dpi=300)
    plt.close()
    print("Saved plot_performance.pdf")


def plot_scalability(df):
    """Generates the scalability violin plot (Ours vs FDDMWR)."""
    print("Generating scalability plot...")
    solvers = [TARGET_ML_SOLVER, "fddmwr"]
    plot_df = df[
        df["solver_id_norm"].isin(solvers) & df["benchmark_name"].isin(["bm"])
    ].copy()

    def get_area(s):
        try:
            parts = s.split("x")
            return int(parts[0]) * int(parts[1])
        except:
            return 0

    sorted_sizes = sorted(plot_df["size"].unique(), key=get_area)
    plot_df["size"] = pd.Categorical(
        plot_df["size"], categories=sorted_sizes, ordered=True
    )

    label_map = {TARGET_ML_SOLVER: "Ours (Single)", "fddmwr": "FDD/MWR"}
    plot_df["solver_label"] = plot_df["solver_id_norm"].map(label_map)

    fig, ax = plt.subplots(figsize=(3.5, 2.0))

    sns.violinplot(
        data=plot_df,
        x="size",
        y="gap",
        hue="solver_label",
        hue_order=["Ours (Single)", "FDD/MWR"],
        split=True,
        inner="quartile",
        palette=["#FFFFFF", "#E0E0E0"],
        cut=0,
        linewidth=0.8,
        ax=ax,
    )

    for i, collection in enumerate(ax.collections):
        if isinstance(collection, mcoll.PolyCollection):
            if i % 2 == 1:
                collection.set_hatch("////")
                collection.set_edgecolor("black")
            else:
                collection.set_edgecolor("black")

    legend_handles = [
        Patch(facecolor="#FFFFFF", edgecolor="black", label="Ours (Single)"),
        Patch(facecolor="#E0E0E0", edgecolor="black", hatch="////", label="FDD/MWR"),
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper right",
        frameon=False,
        ncol=1,
        handletextpad=0.5,
        borderaxespad=0.2,
    )

    ax.yaxis.set_major_formatter(
        FuncFormatter(lambda x, pos: f"{x:.2g}" if x != 0 else "0")
    )
    ax.set_ylabel("Optimality Gap (%)")
    ax.set_xlabel(r"Problem Sizes (Jobs $\times$ Machines)")
    ax.yaxis.grid(True, linestyle=":", alpha=0.5)
    sns.despine()

    formatted_size_labels = [
        s.replace("x", r"$\times$") if "x" in str(s) else s for s in sorted_sizes
    ]
    ax.set_xticks(range(len(sorted_sizes)))
    ax.set_xticklabels(formatted_size_labels, rotation=90)
    plt.xticks(rotation=90)

    plt.tight_layout(pad=0.2)
    plt.savefig("plot_scalability.pdf", format="pdf", bbox_inches="tight", dpi=300)
    plt.close()
    print("Saved plot_scalability.pdf")


def get_train_size(solver_id):
    """Extracts training size from solver ID."""
    try:
        match = re.search(r"(\d+x\d+)_reward", solver_id)
        if match:
            return match.group(1)
    except Exception:
        pass

    return "Unknown"


def plot_generalization(df):
    """Generates generalization line plot."""
    print("Generating generalization plot...")

    plot_df = df[df["solver_type"] == "ml"].copy()
    plot_df = plot_df[plot_df["benchmark_name"].isin(["jh", "sm", "bm"])]
    plot_df["train_size"] = plot_df["solver_id_norm"].apply(get_train_size)

    solver_performance = (
        plot_df.groupby(["solver_id_norm", "train_size"])["gap"].mean().reset_index()
    )
    best_solvers_idx = solver_performance.groupby("train_size")["gap"].idxmin()
    best_solvers = solver_performance.loc[best_solvers_idx, ["solver_id_norm"]]
    plot_df = plot_df[plot_df["solver_id_norm"].isin(best_solvers["solver_id_norm"])]

    heur_df = df[df["solver_type"] == "heuristic"].copy()
    heur_df = heur_df[heur_df["benchmark_name"].isin(["jh", "sm", "bm"])]
    best_heur_df = heur_df[heur_df["solver_id"] == "fddmwr"].copy()
    best_heur_df["train_size"] = "fddmwr"
    plot_df = pd.concat([plot_df, best_heur_df], ignore_index=True)

    plot_df["gap_pct"] = plot_df["gap"] * 100
    plot_df["test_size_prod"] = plot_df["num_jobs"] * plot_df["num_machines"]

    available_sizes = plot_df["train_size"].unique()
    focus_sizes = [
        s
        for s in [
            "10x10",
            "20x20",
            "25x25",
            "30x20",
            "30x10",
            "40x10",
            "50x10",
            "40x20",
            "fddmwr",
        ]
        if s in available_sizes
    ]

    plot_df = plot_df[plot_df["train_size"].isin(focus_sizes)].copy()

    def get_sort_key(s):
        if "x" not in str(s):
            return (99, 0)
        j, m = map(int, s.split("x"))
        ratio = j / m
        is_square = 0 if ratio == 1 else 1
        return (is_square, j * m)

    unique_sizes = plot_df["size"].unique()
    sorted_sizes = sorted(unique_sizes, key=get_sort_key)
    plot_df["size"] = pd.Categorical(
        plot_df["size"], categories=sorted_sizes, ordered=True
    )
    plot_df = plot_df.sort_values("size")

    label_map = {
        "20x20": r"20$\times$20",
        "25x25": r"25$\times$25",
        "30x20": r"30$\times$20",
        "10x10": r"10$\times$10",
        "fddmwr": "FDD/MWR",
        "30x10": r"30$\times$10",
        "40x10": r"40$\times$10",
        "50x10": r"50$\times$10",
        "40x20": r"40$\times$20",
    }
    plot_df["legend_label"] = plot_df["train_size"].map(label_map)

    color_map = {
        label_map["20x20"]: "#000000",
        label_map["25x25"]: "#444444",
        label_map["30x20"]: "#666666",
        label_map["10x10"]: "#888888",
        label_map["fddmwr"]: "#888888",
        label_map["30x10"]: "#CCCCCC",
        label_map["40x10"]: "#CCCCCC",
        label_map["50x10"]: "#CCCCCC",
        label_map["40x20"]: "#CCCCCC",
    }

    style_map = {
        label_map["20x20"]: "",
        label_map["25x25"]: (3, 1, 1, 1),
        label_map["30x20"]: (3, 1),
        label_map["10x10"]: (1, 1),
        label_map["fddmwr"]: (1, 1),
        label_map["30x10"]: (5, 5),
        label_map["40x10"]: (5, 5),
        label_map["50x10"]: (5, 5),
        label_map["40x20"]: (5, 5),
    }

    width_map = {
        label_map["20x20"]: 2.0,
        label_map["25x25"]: 1.5,
        label_map["30x20"]: 1.2,
        label_map["10x10"]: 1.0,
        label_map["fddmwr"]: 1.0,
        label_map["30x10"]: 0.8,
        label_map["40x10"]: 0.8,
        label_map["50x10"]: 0.8,
        label_map["40x20"]: 0.8,
    }

    valid_marker_map = {
        label_map["20x20"]: "o",
        label_map["25x25"]: "^",
        label_map["30x20"]: "D",
        label_map["10x10"]: "s",
        label_map["fddmwr"]: "X",
        label_map["40x20"]: "P",
    }

    fig, ax = plt.subplots(figsize=(3.5, 2.3))

    hero_df = plot_df[plot_df["train_size"] == "20x20"]
    context_df = plot_df[
        plot_df["train_size"].isin(["10x10", "fddmwr", "25x25", "30x20"])
    ]
    noise_df = plot_df[plot_df["train_size"].isin(["30x10", "40x10", "50x10", "40x20"])]

    if not noise_df.empty:
        sns.lineplot(
            data=noise_df,
            x="size",
            y="gap_pct",
            hue="legend_label",
            style="legend_label",
            palette=color_map,
            dashes=style_map,
            size="legend_label",
            sizes=width_map,
            markers=False,
            legend=False,
            zorder=1,
            alpha=0.4,
            ax=ax,
        )

    if not context_df.empty:
        sns.lineplot(
            data=context_df,
            x="size",
            y="gap_pct",
            hue="legend_label",
            style="legend_label",
            palette=color_map,
            dashes=style_map,
            size="legend_label",
            sizes=width_map,
            markers=valid_marker_map,
            legend="brief",
            zorder=2,
            ax=ax,
        )

    if not hero_df.empty:
        sns.lineplot(
            data=hero_df,
            x="size",
            y="gap_pct",
            hue="legend_label",
            style="legend_label",
            palette=color_map,
            dashes=style_map,
            size="legend_label",
            sizes=width_map,
            markers=valid_marker_map,
            err_style="bars",
            errorbar=("ci", 95),
            err_kws={"capsize": 2, "capthick": 1},
            zorder=3,
            legend="brief",
            ax=ax,
        )

    ax.set_xlabel(r"Problem Sizes Jobs $\times$ Machines", fontsize=9)
    ax.set_ylabel("Optimality Gap (%)", fontsize=9)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
    ax.set_yticks([20, 40, 60])
    sns.despine()
    ax.grid(True, linestyle=":", alpha=0.5, color="gray", linewidth=0.5)

    handles, labels = ax.get_legend_handles_labels()
    unique_legend_data = {}
    for handle, label in zip(handles, labels):
        if label in label_map.values() and label not in unique_legend_data:
            unique_legend_data[label] = handle

    sort_order = [
        label_map["20x20"],
        label_map["25x25"],
        label_map["30x20"],
        label_map["10x10"],
        label_map["fddmwr"],
    ]

    final_handles = [
        unique_legend_data[k] for k in sort_order if k in unique_legend_data
    ]
    final_labels = [k for k in sort_order if k in unique_legend_data]

    noise_line = Line2D([0], [0], color="#D0D0D0", lw=1, linestyle="--")
    final_handles.append(noise_line)
    final_labels.append("High Ratio")

    ax.legend(
        final_handles,
        final_labels,
        title=None,
        frameon=False,
        loc="upper left",
        ncol=2,
        columnspacing=1.0,
        handletextpad=0.4,
        handlelength=1.5,
    )

    formatted_size_labels = [
        s.replace("x", r"$\times$") if "x" in str(s) else s for s in sorted_sizes
    ]
    ax.set_xticks(range(len(sorted_sizes)))
    ax.set_xticklabels(formatted_size_labels, rotation=90)

    plt.tight_layout()
    plt.savefig(
        "plot_generalization_per_size.pdf", format="pdf", bbox_inches="tight", dpi=300
    )
    plt.close()
    print("Saved plot_generalization_per_size.pdf")


def plot_ratio_generalization(df):
    """Generates generalization line plot by J/M ratio."""
    print("Generating ratio generalization plot...")

    plot_df = df[df["solver_type"] == "ml"].copy()
    plot_df = plot_df[plot_df["benchmark_name"].isin(["jh", "sm", "bm"])]
    plot_df["train_size"] = plot_df["solver_id_norm"].apply(get_train_size)

    solver_performance = (
        plot_df.groupby(["solver_id_norm", "train_size"])["gap"].mean().reset_index()
    )
    best_solvers_idx = solver_performance.groupby("train_size")["gap"].idxmin()
    best_solvers = solver_performance.loc[best_solvers_idx, ["solver_id_norm"]]
    plot_df = plot_df[plot_df["solver_id_norm"].isin(best_solvers["solver_id_norm"])]

    heur_df = df[df["solver_type"] == "heuristic"].copy()
    heur_df = heur_df[heur_df["benchmark_name"].isin(["jh", "sm", "bm"])]
    best_heur_df = heur_df[heur_df["solver_id"] == "fddmwr"].copy()
    best_heur_df["train_size"] = "fddmwr"
    plot_df = pd.concat([plot_df, best_heur_df], ignore_index=True)

    plot_df["gap_pct"] = plot_df["gap"] * 100
    plot_df["test_size_prod"] = plot_df["num_jobs"] * plot_df["num_machines"]

    available_sizes = plot_df["train_size"].unique()
    focus_sizes = [
        s
        for s in [
            "10x10",
            "20x20",
            "25x25",
            "30x20",
            "30x10",
            "40x10",
            "50x10",
            "40x20",
            "fddmwr",
        ]
        if s in available_sizes
    ]
    plot_df = plot_df[plot_df["train_size"].isin(focus_sizes)].copy()

    def get_ratio_nm(s):
        if "x" not in str(s):
            return None
        n_jobs, m_machines = map(int, s.split("x"))
        return n_jobs / m_machines

    plot_df["ratio_n_m"] = plot_df["size"].apply(get_ratio_nm)
    plot_df = plot_df.dropna(subset=["ratio_n_m"])

    label_map = {
        "20x20": r"20$\times$20",
        "25x25": r"25$\times$25",
        "30x20": r"30$\times$20",
        "10x10": r"10$\times$10",
        "fddmwr": r"FDD/MWR",
        "30x10": r"30$\times$10",
        "40x10": r"40$\times$10",
        "50x10": r"50$\times$10",
        "40x20": r"40$\times$20",
    }
    plot_df["legend_label"] = plot_df["train_size"].map(label_map)

    color_map = {
        label_map["20x20"]: "#000000",
        label_map["25x25"]: "#444444",
        label_map["30x20"]: "#666666",
        label_map["10x10"]: "#888888",
        label_map["fddmwr"]: "#888888",
        label_map["30x10"]: "#CCCCCC",
        label_map["40x10"]: "#CCCCCC",
        label_map["50x10"]: "#CCCCCC",
        label_map["40x20"]: "#CCCCCC",
    }

    style_map = {
        label_map["20x20"]: "",
        label_map["25x25"]: (3, 1, 1, 1),
        label_map["30x20"]: (3, 1),
        label_map["10x10"]: (1, 1),
        label_map["fddmwr"]: (1, 1),
        label_map["30x10"]: (5, 5),
        label_map["40x10"]: (5, 5),
        label_map["50x10"]: (5, 5),
        label_map["40x20"]: (5, 5),
    }

    width_map = {
        label_map["20x20"]: 2.0,
        label_map["25x25"]: 1.5,
        label_map["30x20"]: 1.2,
        label_map["10x10"]: 1.0,
        label_map["fddmwr"]: 1.0,
        label_map["30x10"]: 0.8,
        label_map["40x10"]: 0.8,
        label_map["50x10"]: 0.8,
        label_map["40x20"]: 0.8,
    }

    valid_marker_map = {
        label_map["20x20"]: "o",
        label_map["25x25"]: "^",
        label_map["30x20"]: "D",
        label_map["10x10"]: "s",
        label_map["fddmwr"]: "X",
        label_map["40x20"]: "P",
    }

    fig, ax = plt.subplots(figsize=(3.5, 2.0))

    df_noise = plot_df[plot_df["train_size"].isin(["30x10", "40x10", "50x10", "40x20"])]
    df_context = plot_df[
        plot_df["train_size"].isin(["10x10", "25x25", "30x20", "fddmwr"])
    ]
    df_hero = plot_df[plot_df["train_size"] == "20x20"]

    if not df_noise.empty:
        sns.lineplot(
            data=df_noise,
            x="ratio_n_m",
            y="gap_pct",
            hue="legend_label",
            style="legend_label",
            palette=color_map,
            dashes=style_map,
            size="legend_label",
            sizes=width_map,
            markers=False,
            legend=False,
            zorder=1,
            alpha=0.4,
            ax=ax,
        )

    if not df_context.empty:
        sns.lineplot(
            data=df_context,
            x="ratio_n_m",
            y="gap_pct",
            hue="legend_label",
            style="legend_label",
            palette=color_map,
            dashes=style_map,
            size="legend_label",
            sizes=width_map,
            markers=valid_marker_map,
            legend="brief",
            zorder=2,
            err_style="bars",
            errorbar=("ci", 95),
            err_kws={"capsize": 2, "capthick": 1},
            ax=ax,
        )

    if not df_hero.empty:
        sns.lineplot(
            data=df_hero,
            x="ratio_n_m",
            y="gap_pct",
            hue="legend_label",
            style="legend_label",
            palette=color_map,
            dashes=style_map,
            size="legend_label",
            sizes=width_map,
            markers=valid_marker_map,
            legend="brief",
            zorder=3,
            err_style="bars",
            errorbar=("ci", 95),
            err_kws={"capsize": 2, "capthick": 1},
            ax=ax,
        )

    ax.set_ylabel("Optimality Gap (%)", fontsize=9)
    ax.set_xlabel("Ratio (Jobs $/$ Machines)", fontsize=9)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
    ax.set_yticks([20, 40, 60])

    sns.despine()
    ax.grid(True, linestyle=":", alpha=0.5, color="gray", linewidth=0.5)

    handles, labels = ax.get_legend_handles_labels()
    by_label = {}
    for h, l in zip(handles, labels):
        if l in label_map.values() and l not in by_label:
            by_label[l] = h

    sort_order = [
        label_map["20x20"],
        label_map["25x25"],
        label_map["30x20"],
        label_map["10x10"],
        label_map["fddmwr"],
    ]
    main_handles = [by_label[k] for k in sort_order if k in by_label]
    main_labels = [k for k in sort_order if k in by_label]

    noise_line = Line2D([0], [0], color="#D0D0D0", lw=1, linestyle="--")
    main_handles.append(noise_line)
    main_labels.append("High Ratio")

    ax.legend(
        main_handles,
        main_labels,
        title=None,
        frameon=False,
        loc="upper left",
        ncol=2,
        columnspacing=1.0,
        handletextpad=0.4,
        handlelength=1.5,
    )

    plt.tight_layout()
    plt.savefig(
        "plot_generalization_ratio.pdf", format="pdf", bbox_inches="tight", dpi=300
    )
    plt.close()
    print("Saved plot_generalization_ratio.pdf")


def main():
    print(f"Working directory: {Path.cwd()}")
    configure_plotting("ieee")

    try:
        df, conn = load_data()

        plot_performance_boxplot(df)
        plot_scalability(df)
        plot_generalization(df)
        plot_ratio_generalization(df)

        print("\nAll plots generated successfully.")
    except Exception as e:
        print(f"Error during execution: {e}")
        import traceback

        traceback.print_exc()
    finally:
        if "conn" in locals():
            conn.close()


if __name__ == "__main__":
    main()
