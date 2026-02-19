#!/usr/bin/env python3
"""
Create shareable archives from training runs stored under outputs/.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from omegaconf import OmegaConf

from tools.model_packager import (
    ModelPackagingError,
    build_auto_package_name,
    discover_training_runs,
    package_model_run,
)
from tools.storage.base import ModelStorageBackend, create_storage_backend


def human_readable_size(num_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(num_bytes)
    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "run_dir",
        type=lambda value: Path(value).expanduser().resolve(),
        help="Hydra output folder that contains checkpoints",
    )
    parser.add_argument(
        "--storage-dir",
        type=lambda value: Path(value).expanduser().resolve(),
        default=Path("model_store"),
        help="Directory where the packaged archive will be placed",
    )
    parser.add_argument(
        "--name",
        help="Explicit package name (omit to derive from the run path)",
    )
    parser.add_argument(
        "--prefix-name",
        help="Prefix to add to the package name (useful with --auto)",
    )
    parser.add_argument(
        "--include-logs",
        action="store_true",
        help="Include the run's logs/ folder (TensorBoard)",
    )
    parser.add_argument(
        "--skip-hydra",
        action="store_true",
        help="Do not bundle the .hydra configuration directory",
    )
    parser.add_argument(
        "--extra-file",
        action="append",
        default=[],
        help="Additional file or directory to include (may be used multiple times)",
    )
    parser.add_argument(
        "--notes",
        help="Optional short description stored inside the manifest",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting an existing package with the same name",
    )
    parser.add_argument(
        "--auto",
        action="store_true",
        help="Package every training run found under run_dir recursively",
    )
    parser.add_argument(
        "--upload-backend",
        choices=["azure"],
        help="Optionally upload the packaged files to a remote backend",
    )
    parser.add_argument(
        "--remote-prefix",
        default="",
        help="Remote folder/prefix used when uploading artifacts",
    )
    parser.add_argument(
        "--azure-container",
        help="Azure Blob Storage container name (required when uploading to Azure)",
    )
    parser.add_argument(
        "--azure-connection-string",
        help="Azure Blob connection string (falls back to AZURE_STORAGE_CONNECTION_STRING)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.auto and args.name:
        print("✗ --name cannot be used together with --auto packaging.")
        return 1

    backend = _build_backend(args) if args.upload_backend else None
    extra_files = [Path(p).expanduser().resolve() for p in args.extra_file]

    if args.auto:
        return autopackage_runs(args, backend, extra_files)

    return package_single_run(args, backend, extra_files)


def package_single_run(
    args: argparse.Namespace,
    backend: ModelStorageBackend | None,
    extra_files: list[Path],
) -> int:
    try:
        result = package_model_run(
            args.run_dir,
            storage_dir=args.storage_dir,
            package_name=args.name,
            include_logs=args.include_logs,
            include_hydra=not args.skip_hydra,
            extra_files=extra_files,
            notes=args.notes,
            overwrite=args.overwrite,
        )
    except ModelPackagingError as exc:
        print(f"✗ Packaging failed: {exc}")
        return 1

    _print_result_summary(result, args.notes)

    if backend:
        upload_packaged_result(backend, result)
    return 0


def autopackage_runs(
    args: argparse.Namespace,
    backend: ModelStorageBackend | None,
    extra_files: list[Path],
) -> int:
    base_dir = args.run_dir
    try:
        candidates = discover_training_runs(base_dir)
    except FileNotFoundError as exc:
        print(f"✗ {exc}")
        return 1

    if not candidates:
        print(f"✗ No training runs with checkpoints found under {base_dir}")
        return 1

    successes = 0
    failures = 0
    for run_dir, cfg in candidates:
        package_name = build_auto_package_name(run_dir, base_dir, cfg)
        if args.prefix_name:
            package_name = f"{args.prefix_name}_{package_name}"

        try:
            result = package_model_run(
                run_dir,
                storage_dir=args.storage_dir,
                package_name=package_name,
                include_logs=args.include_logs,
                include_hydra=not args.skip_hydra,
                extra_files=extra_files,
                notes=args.notes,
                overwrite=args.overwrite,
                config_data=_cfg_dict_from_conf(cfg),
            )
        except ModelPackagingError as exc:
            failures += 1
            print(f"✗ {run_dir}: {exc}")
            continue

        successes += 1
        _print_result_summary(result, args.notes)
        if backend:
            upload_packaged_result(backend, result)

    print(f"Completed auto-packaging: {successes} success, {failures} failure(s).")
    return 0 if failures == 0 else 1


def _print_result_summary(result, notes: str | None) -> None:
    print(f"✓ Created archive: {result.archive_path}")
    print(f"  Manifest: {result.manifest_path}")
    print(
        f"  Contents: {result.file_count} files, {human_readable_size(result.total_bytes)}"
    )
    if notes:
        print(f"  Notes: {notes}")


def upload_packaged_result(backend: ModelStorageBackend, result) -> None:
    archive_remote = result.archive_path.name
    manifest_remote = result.manifest_path.name

    backend.upload_file(result.archive_path, archive_remote)
    backend.upload_file(result.manifest_path, manifest_remote)

    print("☁ Uploaded package:")
    print(f"   - {result.archive_path.name} → {archive_remote}")
    print(f"   - {result.manifest_path.name} → {manifest_remote}")


def _build_backend(args: argparse.Namespace) -> ModelStorageBackend:
    backend_name = args.upload_backend
    kwargs = {}
    if backend_name == "azure":
        if not args.azure_container:
            raise SystemExit(
                "--azure-container is required when uploading to Azure Blob Storage."
            )
        kwargs.update(
            {
                "container": args.azure_container,
                "connection_string": args.azure_connection_string,
                "prefix": args.remote_prefix or "",
            }
        )
    return create_storage_backend(backend_name, **kwargs)


def _cfg_dict_from_conf(cfg) -> dict:
    try:
        container = OmegaConf.to_container(cfg, resolve=False)
    except Exception:
        container = None
    if isinstance(container, dict):
        return container
    return {}


if __name__ == "__main__":
    raise SystemExit(main())
