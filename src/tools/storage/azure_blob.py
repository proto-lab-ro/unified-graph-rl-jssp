from __future__ import annotations

import os
from collections.abc import Iterable
from pathlib import Path

from tools.storage.base import RemoteFileInfo


class AzureBlobStorage:
    """
    Azure Blob Storage backend for synchronizing model artifacts.

    Requires the `azure-storage-blob` package. Supply credentials via
    connection string or the AZURE_STORAGE_CONNECTION_STRING environment variable.
    """

    def __init__(
        self,
        *,
        container: str,
        connection_string: str | None = None,
        prefix: str = "",
        overwrite: bool = True,
    ):
        if not container:
            raise ValueError("Azure Blob container name must be provided.")

        if connection_string is None:
            connection_string = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")

        if not connection_string:
            raise ValueError(
                "Azure connection string not provided. "
                "Set AZURE_STORAGE_CONNECTION_STRING or pass --azure-connection-string."
            )

        try:
            from azure.storage.blob import BlobServiceClient
        except ModuleNotFoundError as exc:  # pragma: no cover - import guard
            raise RuntimeError(
                "The 'azure-storage-blob' package is required for Azure uploads."
            ) from exc

        self._service_client = BlobServiceClient.from_connection_string(
            connection_string
        )
        self._container_client = self._service_client.get_container_client(container)
        try:
            self._container_client.create_container()
        except Exception:
            # Container likely already exists or permissions forbid creation.
            pass
        self._base_prefix = prefix.strip("/")
        self._overwrite = overwrite

    # Public API ---------------------------------------------------------
    def upload_file(self, local_path: Path, remote_path: str) -> None:
        local_path = Path(local_path)
        blob_name = self._full_blob_path(remote_path)
        with local_path.open("rb") as data:
            self._container_client.upload_blob(
                name=blob_name,
                data=data,
                overwrite=self._overwrite,
            )

    def download_file(self, remote_path: str, local_path: Path) -> None:
        local_path = Path(local_path)
        blob_name = self._full_blob_path(remote_path)
        downloader = self._container_client.download_blob(blob_name)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        with local_path.open("wb") as handle:
            downloader.readinto(handle)

    def upload_directory(self, local_dir: Path, remote_prefix: str = "") -> None:
        local_dir = Path(local_dir)
        if not local_dir.is_dir():
            raise FileNotFoundError(f"Local directory not found: {local_dir}")

        for file_path in sorted(self._iter_files(local_dir)):
            relative = file_path.relative_to(local_dir).as_posix()
            remote_path = self._join_remote(remote_prefix, relative)
            self.upload_file(file_path, remote_path)

    def download_directory(self, local_dir: Path, remote_prefix: str = "") -> None:
        local_dir = Path(local_dir)
        local_dir.mkdir(parents=True, exist_ok=True)
        prefix = self._full_blob_path(remote_prefix, include_trailing=True)
        blobs = self._container_client.list_blobs(name_starts_with=prefix)
        for blob in blobs:
            blob_name = blob.name
            relative = blob_name[len(prefix) :].lstrip("/")
            if not relative:
                continue
            target_path = local_dir / relative
            target_path.parent.mkdir(parents=True, exist_ok=True)
            downloader = self._container_client.download_blob(blob_name)
            with target_path.open("wb") as handle:
                downloader.readinto(handle)

    def iter_files(self, remote_prefix: str = "") -> Iterable[RemoteFileInfo]:
        prefix = self._full_blob_path(remote_prefix, include_trailing=True)
        base_len = len(prefix)
        blobs = self._container_client.list_blobs(
            name_starts_with=prefix if prefix else None
        )
        for blob in blobs:
            blob_name = blob.name
            if prefix:
                if not blob_name.startswith(prefix):
                    continue
                relative = blob_name[base_len:]
            else:
                relative = blob_name
            relative = relative.lstrip("/")
            if not relative:
                continue
            size = getattr(blob, "size", None)
            checksum = None
            content_settings = getattr(blob, "content_settings", None)
            if content_settings is not None:
                md5_bytes = getattr(content_settings, "content_md5", None)
                if isinstance(md5_bytes, bytes):
                    checksum = md5_bytes.hex()
                elif isinstance(md5_bytes, str):
                    checksum = md5_bytes
            yield RemoteFileInfo(path=relative, size=size, checksum=checksum)

    # Helpers ------------------------------------------------------------
    def _full_blob_path(
        self, remote_path: str, *, include_trailing: bool = False
    ) -> str:
        cleaned = remote_path.replace("\\", "/").strip("/")
        if include_trailing and cleaned:
            cleaned = f"{cleaned}/"
        if self._base_prefix:
            if cleaned:
                return f"{self._base_prefix}/{cleaned}"
            if include_trailing:
                return f"{self._base_prefix}/"
            return self._base_prefix
        return cleaned

    def _join_remote(self, prefix: str, relative: str) -> str:
        prefix = prefix.strip("/")
        if prefix:
            return f"{prefix}/{relative}"
        return relative

    @staticmethod
    def _iter_files(root: Path) -> Iterable[Path]:
        for path in root.rglob("*"):
            if path.is_file():
                yield path
