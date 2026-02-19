#!/usr/bin/env python3
"""
Minimal utility CLI for cleaning up outputs directory.

Features:
- Remove folders with small TensorBoard logs (incomplete runs)
- Remove folders without checkpoints/policy_module_final.pt (failed/incomplete training)
"""

import argparse
import shutil
from pathlib import Path


def find_small_tfevent_files(
    outputs_dir: Path, size_threshold_kb: float = 10.0
) -> list[tuple[Path, Path, float]]:
    """
    Find TensorBoard event files smaller than the threshold.

    Args:
        outputs_dir: Path to the outputs directory
        size_threshold_kb: Size threshold in kilobytes

    Returns:
        List of tuples (event_file_path, datetime_folder_path, size_kb)
    """
    small_files = []

    def is_time_folder(path: Path) -> bool:
        """Check if folder name matches HH-MM-SS pattern"""
        parts = path.name.split("-")
        return len(parts) == 3 and all(p.isdigit() for p in parts)

    def scan_for_tfevent_files(folder: Path) -> None:
        """Recursively scan for tfevents files in logs folders"""
        # Check if this folder contains a logs subfolder with tfevents
        logs_folder = folder / "logs"
        if logs_folder.exists() and logs_folder.is_dir():
            for event_file in logs_folder.glob("*.tfevents.*"):
                size_bytes = event_file.stat().st_size
                size_kb = size_bytes / 1024

                if size_kb < size_threshold_kb:
                    small_files.append((event_file, folder, size_kb))

    # Try multiple patterns to handle all cases:
    # 1. Direct time folders (when outputs_dir is a date folder)
    # 2. Date folders containing time folders (when outputs_dir is root)
    for child in outputs_dir.iterdir():
        if not child.is_dir():
            continue

        # Check if this child has a logs folder directly (time folder pattern)
        if (child / "logs").exists():
            scan_for_tfevent_files(child)
        else:
            # Check subdirectories (might be date folder containing time folders)
            for subchild in child.iterdir():
                if subchild.is_dir() and (subchild / "logs").exists():
                    scan_for_tfevent_files(subchild)

    return small_files


def find_folders_without_final_policy(outputs_dir: Path) -> list[Path]:
    """
    Find all output folders that don't contain a checkpoints/policy_module_final.pt file.

    Args:
        outputs_dir: Path to the outputs directory

    Returns:
        List of folder paths without checkpoints/policy_module_final.pt
    """
    folders_without_policy = []

    # Scan through date and time folders
    for child in outputs_dir.iterdir():
        if not child.is_dir():
            continue

        # Check if this is a time folder with potential model files
        if not (child / ".hydra").exists():  # Not a run folder
            # Check subdirectories (date folders containing time folders)
            for subchild in child.iterdir():
                if subchild.is_dir() and (subchild / ".hydra").exists():
                    # This looks like a run folder
                    if not (
                        subchild / "checkpoints" / "policy_module_final.pt"
                    ).exists():
                        folders_without_policy.append(subchild)
        else:
            # This is a run folder at top level
            if not (child / "checkpoints" / "policy_module_final.pt").exists():
                folders_without_policy.append(child)

    return folders_without_policy


def delete_folders_without_final_policy(
    outputs_dir: Path, dry_run: bool = False
) -> None:
    """
    Delete output folders that don't contain a checkpoints/policy_module_final.pt file.

    Args:
        outputs_dir: Path to the outputs directory
        dry_run: If True, only print what would be deleted without deleting
    """
    folders_to_delete = find_folders_without_final_policy(outputs_dir)

    if not folders_to_delete:
        print("No folders without checkpoints/policy_module_final.pt found.")
        return

    print(
        f"Found {len(folders_to_delete)} folder(s) without checkpoints/policy_module_final.pt:\n"
    )

    for folder in folders_to_delete:
        print(f"📁 {folder.relative_to(outputs_dir)}")

        if dry_run:
            print("   [DRY RUN] Would delete folder\n")
        else:
            try:
                shutil.rmtree(folder)
                print("   ✓ Deleted folder\n")
            except (OSError, PermissionError) as e:
                print(f"   ✗ Error deleting folder: {e}\n")


def delete_small_logs(
    outputs_dir: Path, size_threshold_kb: float = 10.0, dry_run: bool = False
) -> None:
    """
    Delete datetime folders containing small TensorBoard event files.

    Args:
        outputs_dir: Path to the outputs directory
        size_threshold_kb: Size threshold in kilobytes
        dry_run: If True, only print what would be deleted without deleting
    """
    small_files = find_small_tfevent_files(outputs_dir, size_threshold_kb)

    if not small_files:
        print(f"No TensorBoard event files smaller than {size_threshold_kb} KB found.")
        return

    # Group by datetime folder
    folders_to_delete: dict[Path, list[tuple[Path, float]]] = {}
    for event_file, time_folder, size_kb in small_files:
        if time_folder not in folders_to_delete:
            folders_to_delete[time_folder] = []
        folders_to_delete[time_folder].append((event_file, size_kb))

    # Display and delete
    print(
        f"Found {len(small_files)} small TensorBoard event file(s) in {len(folders_to_delete)} folder(s):"
    )
    print(f"Threshold: {size_threshold_kb} KB\n")

    for time_folder, files in folders_to_delete.items():
        print(f"📁 {time_folder.relative_to(outputs_dir)}")
        for event_file, size_kb in files:
            print(f"   └─ {event_file.name} ({size_kb:.2f} KB)")

        if dry_run:
            print("   [DRY RUN] Would delete folder\n")
        else:
            try:
                shutil.rmtree(time_folder)
                print("   ✓ Deleted folder\n")
            except (OSError, PermissionError) as e:
                print(f"   ✗ Error deleting folder: {e}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Clean up outputs directory by removing incomplete or small logs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Dry run (preview what would be deleted)
  python cleanup.py --dry-run

  # Delete logs smaller than 10 KB (default)
  python cleanup.py --small-logs

  # Delete folders without policy_module_final
  python cleanup.py --no-final-policy

  # Delete both (small logs and folders without final policy)
  python cleanup.py --small-logs --no-final-policy

  # Delete logs smaller than 5 KB
  python cleanup.py --small-logs --threshold 5

  # Use a different outputs directory
  python cleanup.py --outputs-dir ./my_outputs --no-final-policy
        """,
    )

    parser.add_argument(
        "--outputs-dir",
        type=lambda p: Path(p).expanduser().resolve(),
        default=Path("outputs"),
        help="Path to the outputs directory (default: outputs)",
    )

    parser.add_argument(
        "--small-logs",
        action="store_true",
        help="Delete folders with small TensorBoard logs",
    )

    parser.add_argument(
        "--no-final-policy",
        action="store_true",
        help="Delete folders without policy_module_final",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=10.0,
        help="Size threshold in KB for small logs (default: 10.0)",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview what would be deleted without actually deleting",
    )

    args = parser.parse_args()

    # If no cleanup mode specified, default to small logs for backward compatibility
    if not args.small_logs and not args.no_final_policy:
        args.small_logs = True

    outputs_dir = args.outputs_dir.resolve()

    if not outputs_dir.exists():
        print(f"Error: Outputs directory not found: {outputs_dir}")
        return 1

    if args.small_logs:
        print("=" * 60)
        print("CLEANING UP SMALL TENSORBOARD LOGS")
        print("=" * 60)
        delete_small_logs(
            outputs_dir=outputs_dir,
            size_threshold_kb=args.threshold,
            dry_run=args.dry_run,
        )

    if args.no_final_policy:
        if args.small_logs:
            print("\n")
        print("=" * 60)
        print("CLEANING UP FOLDERS WITHOUT checkpoints/policy_module_final.pt")
        print("=" * 60)
        delete_folders_without_final_policy(
            outputs_dir=outputs_dir, dry_run=args.dry_run
        )

    return 0


if __name__ == "__main__":
    exit(main())
