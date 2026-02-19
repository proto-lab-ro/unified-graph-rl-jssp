from __future__ import annotations

import importlib
import sys
import types

import pytest

from tools.storage.base import create_storage_backend


class FakeDownloader:
    def __init__(self, content: bytes):
        self._content = content

    def readinto(self, handle):
        handle.write(self._content)


class FakeContainerClient:
    def __init__(self):
        self.blobs: dict[str, bytes] = {}
        self.upload_calls: list[str] = []
        self.download_calls: list[str] = []
        self.created = False
        self.name = ""

    def create_container(self):
        self.created = True

    def upload_blob(self, name: str, data, overwrite: bool):
        self.upload_calls.append(name)
        self.blobs[name] = data.read()

    def download_blob(self, name: str):
        self.download_calls.append(name)
        content = self.blobs.get(name, b"")
        return FakeDownloader(content)

    def list_blobs(self, name_starts_with: str | None = None):
        for blob_name in sorted(self.blobs):
            if name_starts_with and not blob_name.startswith(name_starts_with):
                continue
            yield types.SimpleNamespace(name=blob_name, size=len(self.blobs[blob_name]))


def _install_fake_azure(monkeypatch):
    container = FakeContainerClient()
    blob_module = types.ModuleType("azure.storage.blob")

    class FakeBlobServiceClient:
        def __init__(self):
            self._container = container

        @classmethod
        def from_connection_string(cls, conn: str):
            return cls()

        def get_container_client(self, name: str):
            self._container.name = name
            return self._container

    blob_module.BlobServiceClient = FakeBlobServiceClient
    storage_module = types.ModuleType("azure.storage")
    storage_module.blob = blob_module
    azure_module = types.ModuleType("azure")
    azure_module.storage = storage_module

    monkeypatch.setitem(sys.modules, "azure", azure_module)
    monkeypatch.setitem(sys.modules, "azure.storage", storage_module)
    monkeypatch.setitem(sys.modules, "azure.storage.blob", blob_module)
    sys.modules.pop("tools.storage.azure_blob", None)
    return container


def _load_azure_backend():
    module = importlib.import_module("tools.storage.azure_blob")
    importlib.reload(module)
    return module.AzureBlobStorage


def test_create_storage_backend_unknown():
    with pytest.raises(ValueError):
        create_storage_backend("unknown")


def test_azure_upload_and_download(tmp_path, monkeypatch):
    container = _install_fake_azure(monkeypatch)
    AzureBlobStorage = _load_azure_backend()
    backend = AzureBlobStorage(
        container="models",
        connection_string="UseDevelopmentStorage=true",
    )

    file_path = tmp_path / "pkg.tar.gz"
    file_path.write_bytes(b"package")

    backend.upload_file(file_path, "pkg.tar.gz")
    assert "pkg.tar.gz" in container.blobs

    download_target = tmp_path / "download" / "pkg.tar.gz"
    backend.download_file("pkg.tar.gz", download_target)
    assert download_target.read_bytes() == b"package"


def test_azure_upload_directory_with_prefix(tmp_path, monkeypatch):
    container = _install_fake_azure(monkeypatch)
    AzureBlobStorage = _load_azure_backend()
    backend = AzureBlobStorage(
        container="models",
        connection_string="UseDevelopmentStorage=true",
        prefix="teamA",
    )

    data_dir = tmp_path / "store"
    (data_dir / "sub").mkdir(parents=True)
    (data_dir / "model1.tar.gz").write_bytes(b"m1")
    (data_dir / "sub" / "manifest.json").write_text("{}")

    backend.upload_directory(data_dir, remote_prefix="daily")
    assert "teamA/daily/model1.tar.gz" in container.blobs
    assert "teamA/daily/sub/manifest.json" in container.blobs


def test_storage_factory_returns_azure_backend(monkeypatch):
    _install_fake_azure(monkeypatch)
    backend = create_storage_backend(
        "azure",
        container="models",
        connection_string="UseDevelopmentStorage=true",
    )
    assert backend is not None


def test_azure_iter_files(tmp_path, monkeypatch):
    _install_fake_azure(monkeypatch)
    AzureBlobStorage = _load_azure_backend()
    backend = AzureBlobStorage(
        container="models",
        connection_string="UseDevelopmentStorage=true",
        prefix="teamA",
    )

    file_path = tmp_path / "pkg.tar.gz"
    file_path.write_bytes(b"data")

    backend.upload_file(file_path, "pkg.tar.gz")

    files = list(backend.iter_files())
    assert [info.path for info in files] == ["pkg.tar.gz"]
    assert files[0].size == len(b"data")
