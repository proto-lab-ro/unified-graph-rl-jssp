from __future__ import annotations

import enum
import hashlib
from dataclasses import dataclass
from pathlib import Path

from tools.storage.base import ModelStorageBackend


class SyncDirection(enum.StrEnum):
    UPLOAD = "upload"
    DOWNLOAD = "download"
    BIDIRECTIONAL = "bidirectional"


@dataclass(slots=True)
class SyncStats:
    uploaded: list[str]
    downloaded: list[str]
    skipped: int = 0


def sync_model_store(
    *,
    local_dir: Path,
    backend: ModelStorageBackend,
    direction: SyncDirection = SyncDirection.BIDIRECTIONAL,
    dry_run: bool = False,
) -> SyncStats:
    """
    Synchronize files between local model_store and a remote backend.

    Strategy:
    - Upload files that differ (by size or checksum) or do not exist remotely.
    - Download files that are missing locally when direction allows it.
    - For conflicting files (both exist but sizes differ) in bidirectional
      mode, prefer the local copy.
    """

    local_dir = Path(local_dir)
    local_files = _scan_local(local_dir)
    remote_files = {info.path: info for info in backend.iter_files(remote_prefix="")}

    to_upload = []
    to_download = []
    checksum_cache: dict[str, str] = {}

    def get_local_checksum(rel_path: str) -> str:
        if rel_path not in checksum_cache:
            checksum_cache[rel_path] = _checksum_file(local_dir / rel_path)
        return checksum_cache[rel_path]

    if direction in {SyncDirection.UPLOAD, SyncDirection.BIDIRECTIONAL}:
        for rel_path, size in local_files.items():
            remote_info = remote_files.get(rel_path)
            if remote_info is None:
                to_upload.append(rel_path)
                continue

            remote_size = remote_info.size
            if remote_size != size:
                to_upload.append(rel_path)
                continue

            if remote_info.checksum:
                local_checksum = get_local_checksum(rel_path)
                if local_checksum != remote_info.checksum:
                    to_upload.append(rel_path)

    if direction in {SyncDirection.DOWNLOAD, SyncDirection.BIDIRECTIONAL}:
        for rel_path, remote_info in remote_files.items():
            local_size = local_files.get(rel_path)
            if local_size is None:
                to_download.append(rel_path)
                continue

            if direction != SyncDirection.DOWNLOAD:
                continue

            remote_size = remote_info.size
            if remote_size is not None and local_size != remote_size:
                to_download.append(rel_path)
                continue

            if remote_info.checksum:
                local_checksum = get_local_checksum(rel_path)
                if local_checksum != remote_info.checksum:
                    to_download.append(rel_path)

    if dry_run:
        return SyncStats(uploaded=to_upload, downloaded=to_download, skipped=0)

    for rel_path in to_upload:
        backend.upload_file(local_dir / rel_path, rel_path)

    for rel_path in to_download:
        backend.download_file(rel_path, local_dir / rel_path)

    skipped = (
        len(local_files)
        + len(remote_files)
        - len(set(to_upload))
        - len(set(to_download))
    )
    return SyncStats(uploaded=to_upload, downloaded=to_download, skipped=skipped)


def _scan_local(root: Path) -> dict[str, int]:
    files: dict[str, int] = {}
    if not root.exists():
        return files
    for path in root.rglob("*"):
        if path.is_file():
            rel_path = path.relative_to(root).as_posix()
            files[rel_path] = path.stat().st_size
    return files


def _checksum_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()
