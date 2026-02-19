"""Tiny model store helper utilities (KIS/DRY).

Features kept minimal:
- Load/save JSON manifests from ``model_store``.
- Dot-path helpers for safe lookups.
- Rebuild a small index (JSON cache) for quick filtering.
- Query with a simple expression language using ``get("a.b.c")`` and ``has("a.b")``.

All logic is stdlib-only to avoid extra dependencies.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tools.model_packager import DEFAULT_STORAGE_DIR


Manifest = dict[str, Any]

INDEX_FILENAME = "index.json"


@dataclass(slots=True)
class IndexEntry:
    model_id: str
    manifest_path: Path
    manifest: Manifest

    def as_row(self, columns: Sequence[str]) -> dict[str, Any]:
        return {col: dot_get(self.manifest, col) for col in columns}


# -----------------------------
# Manifest IO
# -----------------------------


def manifest_path(
    model_id_or_path: str | os.PathLike[str], *, storage_dir: Path | str | None = None
) -> Path:
    path = Path(model_id_or_path)
    if path.suffix == ".json" and path.exists():
        return path
    base = Path(storage_dir) if storage_dir else DEFAULT_STORAGE_DIR
    base = base.expanduser().resolve()
    candidate = base / f"{path}.manifest.json"
    return candidate


def load_manifest(
    model_id_or_path: str | os.PathLike[str], *, storage_dir: Path | str | None = None
) -> Manifest:
    path = manifest_path(model_id_or_path, storage_dir=storage_dir)
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if "package_name" not in data:
        data.setdefault("package_name", path.stem.replace(".manifest", ""))
    return data


def save_manifest(
    model_id: str, manifest: Manifest, *, storage_dir: Path | str | None = None
) -> Path:
    path = manifest_path(model_id, storage_dir=storage_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=False), encoding="utf-8")
    return path


# -----------------------------
# Dot-path helpers
# -----------------------------


def dot_get(obj: Any, path: str, default: Any | None = None) -> Any:
    """Lookup dotted paths in nested dict/list structures."""
    parts = [p for p in path.split(".") if p]
    current: Any = obj
    for part in parts:
        if isinstance(current, dict):
            if part not in current:
                return default
            current = current[part]
        elif isinstance(current, list):
            try:
                idx = int(part)
            except ValueError:
                return default
            if idx < 0 or idx >= len(current):
                return default
            current = current[idx]
        else:
            return default
    return current


def dot_set(obj: dict[str, Any], path: str, value: Any) -> None:
    parts = [p for p in path.split(".") if p]
    current: Any = obj
    for part in parts[:-1]:
        if isinstance(current, dict):
            current = current.setdefault(part, {})
        elif isinstance(current, list):
            idx = int(part)
            while len(current) <= idx:
                current.append({})
            if not isinstance(current[idx], dict):
                current[idx] = {}
            current = current[idx]
        else:
            raise TypeError(f"Cannot set path '{path}' on non-container")
    if isinstance(current, dict):
        current[parts[-1]] = value
    elif isinstance(current, list):
        idx = int(parts[-1])
        while len(current) <= idx:
            current.append(None)
        current[idx] = value
    else:
        raise TypeError(f"Cannot set path '{path}' on non-container")


# -----------------------------
# Indexing and querying
# -----------------------------


def iter_manifest_files(storage_dir: Path | str | None = None) -> Iterator[Path]:
    base = Path(storage_dir) if storage_dir else DEFAULT_STORAGE_DIR
    if not base.exists():
        return iter(())
    return (p for p in base.glob("*.manifest.json") if p.is_file())


def rebuild_index(*, storage_dir: Path | str | None = None) -> Path:
    entries: list[IndexEntry] = []
    for path in iter_manifest_files(storage_dir):
        try:
            manifest = load_manifest(path)
        except Exception:
            continue
        model_id = path.name.replace(".manifest.json", "")
        entries.append(
            IndexEntry(model_id=model_id, manifest_path=path, manifest=manifest)
        )
    index_payload = [
        {
            "model_id": e.model_id,
            "manifest_path": str(e.manifest_path),
            "manifest": e.manifest,
        }
        for e in entries
    ]
    base = Path(storage_dir) if storage_dir else DEFAULT_STORAGE_DIR
    base = base.expanduser().resolve()
    index_path = base / INDEX_FILENAME
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(json.dumps(index_payload, indent=2), encoding="utf-8")
    return index_path


def load_index(*, storage_dir: Path | str | None = None) -> list[IndexEntry]:
    base = Path(storage_dir) if storage_dir else DEFAULT_STORAGE_DIR
    index_path = base / INDEX_FILENAME
    if not index_path.exists():
        rebuild_index(storage_dir=storage_dir)
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    entries: list[IndexEntry] = []
    for item in payload:
        entries.append(
            IndexEntry(
                model_id=item["model_id"],
                manifest_path=Path(item["manifest_path"]),
                manifest=item["manifest"],
            )
        )
    return entries


def query_index(
    entries: Iterable[IndexEntry],
    *,
    filter_expr: str | None = None,
    sort: str | None = None,
    limit: int | None = None,
) -> list[IndexEntry]:
    filtered = [e for e in entries if _matches(e.manifest, filter_expr)]
    if sort:
        reverse = sort.startswith("-")
        key_path = sort[1:] if reverse else sort
        filtered.sort(
            key=lambda e: _sortable(dot_get(e.manifest, key_path)), reverse=reverse
        )
    if limit is not None:
        filtered = filtered[:limit]
    return filtered


def _matches(manifest: Manifest, expr: str | None) -> bool:
    if not expr:
        return True
    env: dict[str, Any] = {
        "get": lambda path, default=None: dot_get(manifest, path, default),
        "has": lambda path: dot_get(manifest, path, None) is not None,
        "contains_path": lambda path, value: any(
            dot_get(item, path) == value
            for item in manifest.get("artifacts", [])
            if isinstance(item, dict)
        ),
        "manifest": manifest,
    }
    try:
        return bool(eval(expr, {"__builtins__": {}}, env))
    except Exception:
        return False


def _sortable(value: Any) -> Any:
    if value is None:
        return (1, None)
    if isinstance(value, (int, float, str, bool)):
        return (0, value)
    return (0, str(value))


# -----------------------------
# Benchmark helpers
# -----------------------------


def upsert_benchmarks(
    manifest: Manifest,
    aggregates: dict[str, dict[str, float | int]],
    *,
    aggregated_at: datetime | None = None,
) -> Manifest:
    bench = manifest.setdefault("benchmarks", {})
    bench["aggregates"] = aggregates
    bench["aggregated_at_utc"] = (aggregated_at or datetime.now(UTC)).isoformat(
        timespec="seconds"
    )
    return manifest


# -----------------------------
# Model deletion
# -----------------------------


def delete_model(
    model_id: str, *, storage_dir: Path | str | None = None
) -> tuple[bool, str]:
    """
    Delete a model and its archive from the store.

    Returns:
        (success, message) tuple indicating outcome.
    """
    base = Path(storage_dir) if storage_dir else DEFAULT_STORAGE_DIR
    base = base.expanduser().resolve()

    manifest_path_obj = base / f"{model_id}.manifest.json"
    archive_path = base / f"{model_id}.tar.gz"

    if not manifest_path_obj.exists() and not archive_path.exists():
        return False, f"Model '{model_id}' not found in {base}"

    deleted_files: list[str] = []
    for path in [manifest_path_obj, archive_path]:
        if path.exists():
            try:
                path.unlink()
                deleted_files.append(path.name)
            except OSError as e:
                return False, f"Failed to delete {path.name}: {e}"

    return True, f"Deleted model '{model_id}': {', '.join(deleted_files)}"


__all__ = [
    "IndexEntry",
    "load_manifest",
    "save_manifest",
    "dot_get",
    "dot_set",
    "rebuild_index",
    "load_index",
    "query_index",
    "upsert_benchmarks",
    "delete_model",
]
