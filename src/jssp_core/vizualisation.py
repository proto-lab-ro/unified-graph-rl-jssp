import matplotlib.animation as animation
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

from jssp_core.schedule import Schedule


def plot_gantt_chart(schedule: Schedule, show=True, path=None):
    """
    Plot Gantt chart from either a JSSPEnv or Schedule object

    Args:
        env_or_schedule: Either a JSSPEnv instance or Schedule instance
    """
    if isinstance(schedule, Schedule):
        schedule = schedule
    else:
        raise ValueError("Input must be Schedule instance")

    gantt_data = schedule.get_gantt_data()

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = plt.cm.tab10.colors  # color palette for up to 10 jobs

    for entry in gantt_data:
        job_id = entry["job_id"]
        op_id = entry["op_id"]
        machine = entry["machine"]
        start_time = entry["start_time"]
        duration = entry["duration"]

        color = colors[job_id % len(colors)]
        ax.barh(
            y=machine,
            left=start_time,
            width=duration,
            color=color,
            edgecolor="black",
            height=0.6,
        )
        ax.text(
            start_time + duration / 2,
            machine,
            f"J{job_id}O{op_id}",
            va="center",
            ha="center",
            color="white",
            fontsize=9,
        )

    # Format chart
    ax.set_yticks(range(schedule.num_machines))
    ax.set_yticklabels([f"Machine {m}" for m in range(schedule.num_machines)])
    ax.set_xlabel("Time")
    ax.set_title(f"JSSP Gantt Chart (Makespan: {schedule.get_makespan():.1f})")
    ax.grid(True, axis="x", linestyle="--", alpha=0.5)

    # Add mean and max CLBS lines
    try:
        clbs = list(schedule.get_operation_lower_bounds().values())
        if clbs:
            max_clb = max(clbs)
            mean_clb = sum(clbs) / len(clbs)

            max_line = ax.axvline(
                x=max_clb,
                color="red",
                linestyle="--",
                linewidth=2,
                label=f"Max CLB ({max_clb:.1f})",
            )
            mean_line = ax.axvline(
                x=mean_clb,
                color="blue",
                linestyle="-.",
                linewidth=2,
                label=f"Mean CLB ({mean_clb:.1f})",
            )
    except Exception as e:
        print(f"Could not calculate CLB stats: {e}")
        max_line = None
        mean_line = None

    # Legend
    patches = [
        mpatches.Patch(color=colors[j % len(colors)], label=f"Job {j}")
        for j in range(schedule.num_jobs)
    ]

    # Add lines to legend if they exist
    handles = patches
    if "max_line" in locals() and max_line:
        handles.append(max_line)
    if "mean_line" in locals() and mean_line:
        handles.append(mean_line)

    ax.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.tight_layout()
    if show:
        plt.show()

    if path:
        print(f"Saving Gantt chart to {path}")
        plt.savefig(path, bbox_inches="tight")


def make_gantt_animation(
    schedule: Schedule, interval=200, repeat=True, show=True, save_path=None
):
    """
    Create an animation of the Gantt chart for a given schedule.
    Each operation is displayed in sequence based on their start time.
    Includes dynamic Mean and Max CLB (Conditional Lower Bound) lines.

    Args:
        schedule: Schedule instance containing the operations to visualize.
        interval: Time between frames in milliseconds (default: 200ms)
        repeat: Whether to repeat the animation (default: True)
        show: Whether to display the animation (default: True)
        save_path: Optional path to save the animation (e.g., 'animation.gif' or 'animation.mp4')
                   Requires ffmpeg or imagemagick for saving.

    Returns:
        matplotlib FuncAnimation object
    """

    # Pre-calculate frames data to simulate the scheduling process
    # We reconstruct the schedule step-by-step to get accurate CLB values at each step
    operations = list(schedule.get_all_scheduled_operations())
    operations.sort(key=lambda x: x.start_time)

    # Temporary schedule to track state evolution
    temp_schedule = Schedule(schedule.instance)
    frames_data = []

    print("Pre-calculating animation frames...")
    for op in operations:
        # Manually add operation to preserve exact timing of original schedule
        # (avoiding re-calculation of start times which might differ if greedy)
        temp_schedule.scheduled[(op.operation.job_id, op.operation.op_id)] = (
            op.start_time
        )

        # Calculate CLB stats for this step
        try:
            clbs = list(temp_schedule.get_operation_lower_bounds().values())
            if clbs:
                max_clb = max(clbs)
                mean_clb = sum(clbs) / len(clbs)
            else:
                max_clb = 0
                mean_clb = 0
        except Exception:
            max_clb = 0
            mean_clb = 0

        current_makespan = 0
        if temp_schedule.scheduled:
            # Calculate makespan from currently scheduled ops
            current_makespan = max(
                t + schedule.instance[j][o][1]
                for (j, o), t in temp_schedule.scheduled.items()
            )

        frames_data.append(
            {
                "op": op,
                "max_clb": max_clb,
                "mean_clb": mean_clb,
                "makespan": current_makespan,
                "count": len(temp_schedule.scheduled),
            }
        )

    # Set up the figure and axis
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = plt.cm.tab10.colors  # color palette for up to 10 jobs

    def init():
        """Initialize the animation"""
        ax.clear()
        ax.set_yticks(range(schedule.num_machines))
        ax.set_yticklabels([f"Machine {m}" for m in range(schedule.num_machines)])
        ax.set_xlabel("Time")
        ax.set_title(
            f"JSSP Gantt Chart Animation (Final Makespan: {schedule.get_makespan():.1f})"
        )
        ax.grid(True, axis="x", linestyle="--", alpha=0.5)

        # Set axis limits
        max_time = schedule.get_makespan()
        ax.set_xlim(0, max(max_time * 1.1, 1))  # Ensure non-zero width
        ax.set_ylim(-0.5, schedule.num_machines - 0.5)

        return []

    def animate(frame):
        """Animation function - renders the frame"""
        ax.clear()

        # Get data for this frame
        data = frames_data[frame]

        # Set up the plot again
        ax.set_yticks(range(schedule.num_machines))
        ax.set_yticklabels([f"Machine {m}" for m in range(schedule.num_machines)])
        ax.set_xlabel("Time")
        ax.grid(True, axis="x", linestyle="--", alpha=0.5)

        title = (
            f"Step {data['count']}/{len(operations)} | Makespan: {data['makespan']:.1f}"
        )
        ax.set_title(title)

        # Convert simple list of ops to display
        # We need to show all operations up to this frame
        ops_to_show = operations[: frame + 1]

        # Draw operations
        for i, op in enumerate(ops_to_show):
            job_id = op.operation.job_id
            op_id = op.operation.op_id
            machine = op.operation.machine
            start_time = op.start_time
            duration = op.operation.duration

            color = colors[job_id % len(colors)]

            # Highlight the most recently added operation
            alpha = 1.0 if i == frame else 0.8
            edgecolor = "red" if i == frame else "black"
            linewidth = 2 if i == frame else 1

            # Draw the bar
            ax.barh(
                y=machine,
                left=start_time,
                width=duration,
                color=color,
                edgecolor=edgecolor,
                linewidth=linewidth,
                height=0.6,
                alpha=alpha,
            )

            # Add text label
            ax.text(
                start_time + duration / 2,
                machine,
                f"J{job_id}O{op_id}",
                va="center",
                ha="center",
                color="white",
                fontsize=9,
                weight="bold",
            )

        # Plot CLB Lines
        max_clb = data["max_clb"]
        mean_clb = data["mean_clb"]

        max_line = ax.axvline(
            x=max_clb,
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Max CLB ({max_clb:.1f})",
        )
        mean_line = ax.axvline(
            x=mean_clb,
            color="blue",
            linestyle="-.",
            linewidth=2,
            label=f"Mean CLB ({mean_clb:.1f})",
        )

        # Set axis limits
        final_makespan = schedule.get_makespan()
        ax.set_xlim(0, max(final_makespan * 1.1, 10))
        ax.set_ylim(-0.5, schedule.num_machines - 0.5)

        # Add legend
        shown_jobs = set(op.operation.job_id for op in ops_to_show)
        patches = [
            mpatches.Patch(color=colors[j % len(colors)], label=f"Job {j}")
            for j in sorted(shown_jobs)
        ]

        handles = patches + [max_line, mean_line]
        ax.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc="upper left")

        return []

    # Create the animation
    anim = animation.FuncAnimation(
        fig,
        animate,
        init_func=init,
        frames=len(frames_data),
        interval=interval,
        repeat=repeat,
        blit=False,
    )

    plt.tight_layout()

    if save_path:
        print(f"Saving animation to {save_path}...")
        try:
            if save_path.endswith(".gif"):
                anim.save(save_path, writer="pillow")
            else:
                # Requires ffmpeg
                anim.save(save_path, writer="ffmpeg")
            print("Animation saved successfully.")
        except Exception as e:
            print(f"Failed to save animation: {e}")

    if show:
        print("Showing Gantt chart animation...")
        plt.show()
        print("Close the window to exit.")

    return anim


if __name__ == "__main__":
    from jssp_core.instances import JSSPInstance
    from jssp_core.schedule import Schedule

    # Example usage
    instance = JSSPInstance.from_file("./jssp_instances/ft06")
    schedule = Schedule(instance)
    # Manually create a simple schedule for demonstration
    while not schedule.is_complete():
        for job_id in range(schedule.num_operations):
            schedule.schedule_job(job_id)

    plot_gantt_chart(schedule, show=True)
