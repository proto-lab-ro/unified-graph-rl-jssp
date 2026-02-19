from __future__ import annotations

import hashlib
from pathlib import Path

from tools.model_sync import SyncDirection, sync_model_store
from tools.storage.base import ModelStorageBackend, RemoteFileInfo


class DummyBackend(ModelStorageBackend):
    def __init__(self):
        self.files: dict[str, bytes] = {}
        self.upload_log: list[str] = []
        self.download_log: list[str] = []

    def upload_file(self, local_path: Path, remote_path: str) -> None:
        self.upload_log.append(remote_path)
        self.files[remote_path] = local_path.read_bytes()

    def download_file(self, remote_path: str, local_path: Path) -> None:
        self.download_log.append(remote_path)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        local_path.write_bytes(self.files[remote_path])

    def upload_directory(self, local_dir: Path, remote_prefix: str = "") -> None:
        raise NotImplementedError

    def download_directory(self, local_dir: Path, remote_prefix: str = "") -> None:
        raise NotImplementedError

    def iter_files(self, remote_prefix: str = ""):
        for path, data in self.files.items():
            checksum = hashlib.md5(data).hexdigest()
            yield RemoteFileInfo(path=path, size=len(data), checksum=checksum)


def test_sync_upload_only(tmp_path):
    local_dir = tmp_path / "model_store"
    (local_dir / "a").mkdir(parents=True)
    (local_dir / "a" / "file1").write_text("hello")
    (local_dir / "b.txt").write_text("world")

    backend = DummyBackend()

    stats = sync_model_store(
        local_dir=local_dir,
        backend=backend,
        direction=SyncDirection.UPLOAD,
    )

    assert set(stats.uploaded) == {"a/file1", "b.txt"}
    assert stats.downloaded == []
    assert backend.files["a/file1"] == b"hello"
    assert backend.files["b.txt"] == b"world"


def test_sync_download_missing_local(tmp_path):
    local_dir = tmp_path / "model_store"
    backend = DummyBackend()
    backend.files["model.tar.gz"] = b"pkg"

    stats = sync_model_store(
        local_dir=local_dir,
        backend=backend,
        direction=SyncDirection.DOWNLOAD,
    )

    assert stats.downloaded == ["model.tar.gz"]
    assert (local_dir / "model.tar.gz").read_bytes() == b"pkg"


def test_sync_dry_run(tmp_path):
    local_dir = tmp_path / "model_store"
    (local_dir / "model.tar.gz").parent.mkdir(parents=True, exist_ok=True)
    (local_dir / "model.tar.gz").write_bytes(b"abc")

    backend = DummyBackend()
    backend.files["model.tar.gz"] = b"def"

    stats = sync_model_store(
        local_dir=local_dir,
        backend=backend,
        direction=SyncDirection.BIDIRECTIONAL,
        dry_run=True,
    )

    # Local version should win for uploads; remote missing locally should be downloaded.
    assert stats.uploaded == ["model.tar.gz"]
    assert stats.downloaded == []
    assert backend.files["model.tar.gz"] == b"def"  # No changes because dry run
