from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable


@dataclass(slots=True)
class RemoteFileInfo:
    path: str
    size: int | None = None
    checksum: str | None = None


@runtime_checkable
class ModelStorageBackend(Protocol):
    """Minimal interface for pushing/pulling model artifacts."""

    def upload_file(self, local_path: Path, remote_path: str) -> None: ...

    def download_file(self, remote_path: str, local_path: Path) -> None: ...

    def upload_directory(self, local_dir: Path, remote_prefix: str = "") -> None: ...

    def download_directory(self, local_dir: Path, remote_prefix: str = "") -> None: ...

    def iter_files(self, remote_prefix: str = "") -> Iterable[RemoteFileInfo]: ...


def create_storage_backend(name: str, **backend_kwargs: Any) -> ModelStorageBackend:
    """
    Factory helper for instantiating storage backends.

    Args:
        name: Backend identifier (e.g., "azure").
        backend_kwargs: Keyword arguments forwarded to the specific backend.
    """
    normalized = name.lower().strip()
    if normalized == "azure":
        from tools.storage.azure_blob import AzureBlobStorage

        return AzureBlobStorage(**backend_kwargs)

    raise ValueError(f"Unsupported storage backend: {name}")
