#!/usr/bin/env python3
"""
Synchronize the local model_store directory with remote storage backends.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from tools.model_sync import SyncDirection, sync_model_store
from tools.storage.base import create_storage_backend


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "action",
        choices=["upload", "download", "sync"],
        help="Upload, download, or bidirectionally sync with the remote backend",
    )
    parser.add_argument(
        "--backend",
        required=True,
        choices=["azure"],
        help="Storage backend to use",
    )
    parser.add_argument(
        "--local-dir",
        type=lambda value: Path(value).expanduser().resolve(),
        default=Path("model_store"),
        help="Local directory that stores packaged models",
    )
    parser.add_argument(
        "--remote-prefix",
        default="",
        help="Remote folder/prefix inside the backend container",
    )
    parser.add_argument(
        "--azure-container",
        help="Azure Blob Storage container (required for the azure backend)",
    )
    parser.add_argument(
        "--azure-connection-string",
        help="Azure connection string (falls back to AZURE_STORAGE_CONNECTION_STRING)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show planned actions without transferring files",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    backend = create_storage_backend(
        args.backend,
        **_backend_kwargs(args),
    )

    direction = SyncDirection.BIDIRECTIONAL
    if args.action == "upload":
        direction = SyncDirection.UPLOAD
    elif args.action == "download":
        direction = SyncDirection.DOWNLOAD

    stats = sync_model_store(
        local_dir=args.local_dir,
        backend=backend,
        direction=direction,
        dry_run=args.dry_run,
    )

    action_label = {
        SyncDirection.UPLOAD: "Upload-only",
        SyncDirection.DOWNLOAD: "Download-only",
        SyncDirection.BIDIRECTIONAL: "Bidirectional sync",
    }[direction]

    print(f"{'DRY RUN: ' if args.dry_run else ''}{action_label} results")
    print(f"  Uploaded: {len(stats.uploaded)} file(s)")
    print(f"  Downloaded: {len(stats.downloaded)} file(s)")
    if stats.uploaded:
        print("   → " + ", ".join(stats.uploaded))
    if stats.downloaded:
        print("   ← " + ", ".join(stats.downloaded))

    return 0


def _backend_kwargs(args: argparse.Namespace) -> dict:
    if args.backend == "azure":
        if not args.azure_container:
            raise SystemExit(
                "--azure-container is required when using the Azure backend."
            )
        return {
            "container": args.azure_container,
            "connection_string": args.azure_connection_string,
            "prefix": args.remote_prefix or "",
        }
    return {}


if __name__ == "__main__":
    raise SystemExit(main())
