"""
Utilities to package trained model runs into portable archives for sharing.

Typical usage from Python:

    from pathlib import Path
    from tools.model_packager import package_model_run

    result = package_model_run(Path("outputs/2025-11-18/17-19-01"))
    print(result.archive_path)
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import re
import shutil
import subprocess
import tarfile
import tempfile
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from jssp_core.solver.base import JSSPSolverBase


DEFAULT_STORAGE_DIR = Path("model_store")


class ModelPackagingError(RuntimeError):
    """Raised when packaging cannot complete."""


@dataclass(slots=True)
class PackageResult:
    """Information about a packaged archive."""

    archive_path: Path
    manifest_path: Path
    manifest: dict
    file_count: int
    total_bytes: int
    package_name: str


def package_model_run(
    run_dir: Path | str,
    *,
    storage_dir: Path | str | None = None,
    package_name: str | None = None,
    include_logs: bool = False,
    include_hydra: bool = True,
    extra_files: Sequence[Path | str] | None = None,
    notes: str | None = None,
    overwrite: bool = False,
    config_data: dict | None = None,
) -> PackageResult:
    """
    Package a single training run directory into a compressed archive.

    Args:
        run_dir: Path to the Hydra output folder that contains checkpoints.
        storage_dir: Where packaged archives should be saved (default: model_store).
        package_name: Optional explicit archive name (without extension).
        include_logs: Whether to include the run's `logs/` directory (TensorBoard).
        include_hydra: Whether to bundle `.hydra` configs for reproducibility.
        extra_files: Additional files or directories to include under `extras/`.
        notes: Optional free-form description stored in the manifest.
        overwrite: Allow replacing an existing package of the same name.
    """

    run_path = Path(run_dir).expanduser().resolve()
    if not run_path.is_dir():
        raise ModelPackagingError(f"Run directory does not exist: {run_path}")

    checkpoints_dir = run_path / "checkpoints"
    if not checkpoints_dir.exists():
        raise ModelPackagingError(
            f"No checkpoints directory found in run folder: {run_path}"
        )

    best_model_candidates = list(checkpoints_dir.glob("policy_module_final.pt"))
    if not best_model_candidates:
        raise ModelPackagingError(
            f"No final model (policy_module_final.pt) found in {checkpoints_dir}"
        )

    dest_dir = Path(storage_dir) if storage_dir else DEFAULT_STORAGE_DIR
    dest_dir = dest_dir.expanduser().resolve()
    dest_dir.mkdir(parents=True, exist_ok=True)

    archive_name = (
        _sanitize_name(package_name) if package_name else _derive_package_name(run_path)
    )
    archive_path = dest_dir / f"{archive_name}.tar.gz"
    manifest_path = dest_dir / f"{archive_name}.manifest.json"
    if not overwrite and (archive_path.exists() or manifest_path.exists()):
        raise ModelPackagingError(
            f"Package '{archive_name}' already exists in {dest_dir}. "
            "Choose a different name or pass overwrite=True."
        )

    extras: list[Path] = []
    if extra_files:
        for extra in extra_files:
            extra_path = Path(extra).expanduser().resolve()
            if not extra_path.exists():
                raise ModelPackagingError(f"Extra path does not exist: {extra_path}")
            extras.append(extra_path)

    if config_data is None:
        try:
            run_cfg = load_run_config(run_path)
            config_data = _cfg_to_dict(run_cfg)
        except FileNotFoundError:
            config_data = None

    with tempfile.TemporaryDirectory(prefix="model_package_") as tmp_dir:
        staging_root = Path(tmp_dir) / archive_name
        _populate_staging_directory(
            staging_root=staging_root,
            run_dir=run_path,
            include_logs=include_logs,
            include_hydra=include_hydra,
            extras=extras,
        )

        manifest = _build_manifest(
            staging_root=staging_root,
            run_dir=run_path,
            include_logs=include_logs,
            include_hydra=include_hydra,
            notes=notes,
            config_snapshot=config_data,
        )
        manifest_file = staging_root / "manifest.json"
        manifest_file.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        with tarfile.open(archive_path, "w:gz") as tar:
            tar.add(staging_root, arcname=archive_name)

        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    file_count = sum(1 for item in manifest["artifacts"] if item["type"] == "file")
    total_bytes = sum(item["size_bytes"] for item in manifest["artifacts"])
    return PackageResult(
        archive_path=archive_path,
        manifest_path=manifest_path,
        manifest=manifest,
        file_count=file_count,
        total_bytes=total_bytes,
        package_name=archive_name,
    )


def _populate_staging_directory(
    *,
    staging_root: Path,
    run_dir: Path,
    include_logs: bool,
    include_hydra: bool,
    extras: Sequence[Path],
) -> None:
    if staging_root.exists():
        shutil.rmtree(staging_root)
    staging_root.mkdir(parents=True, exist_ok=True)

    artifacts_dir = staging_root / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_src = run_dir / "checkpoints"
    _copy_path(checkpoints_src, artifacts_dir / "checkpoints")

    logs_dest = staging_root / "logs"
    log_files = sorted(run_dir.glob("*.log"))
    if log_files:
        logs_dest.mkdir(parents=True, exist_ok=True)
        for log_file in log_files:
            shutil.copy2(log_file, logs_dest / log_file.name)

    if include_logs:
        tensorboard_src = run_dir / "logs"
        if tensorboard_src.exists():
            logs_dest.mkdir(parents=True, exist_ok=True)
            _copy_path(tensorboard_src, logs_dest / "tensorboard")

    if include_hydra:
        hydra_src = run_dir / ".hydra"
        if hydra_src.exists():
            configs_dir = staging_root / "configs" / "hydra"
            _copy_path(hydra_src, configs_dir)

    if extras:
        extras_dest = staging_root / "extras"
        extras_dest.mkdir(parents=True, exist_ok=True)
        for src in extras:
            dest = extras_dest / src.name
            _copy_path(src, dest)


def _copy_path(src: Path, dest: Path) -> None:
    if src.is_dir():
        shutil.copytree(src, dest, dirs_exist_ok=True)
    else:
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)


def _build_manifest(
    *,
    staging_root: Path,
    run_dir: Path,
    include_logs: bool,
    include_hydra: bool,
    notes: str | None,
    config_snapshot: dict | None,
) -> dict:
    artifacts: list[dict] = []
    for path in sorted(staging_root.rglob("*")):
        if not path.is_file():
            continue
        rel_path = path.relative_to(staging_root).as_posix()
        if rel_path == "manifest.json":
            continue
        artifacts.append(
            {
                "type": "file",
                "path": rel_path,
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )

    git_info = _git_metadata(run_dir)
    summary = {
        "package_name": staging_root.name,
        "created_at_utc": dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "source_run_dir": str(run_dir),
        "options": {
            "include_logs": include_logs,
            "include_hydra": include_hydra,
        },
        "git": git_info,
        "artifacts": artifacts,
        "config": config_snapshot,
    }
    if notes:
        summary["notes"] = notes
    return summary


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_metadata(start: Path) -> dict:
    repo_root = _find_git_root(start)
    if repo_root is None:
        return {"commit": None, "branch": None, "is_dirty": None}

    commit = _run_git(["rev-parse", "HEAD"], cwd=repo_root)
    branch = _run_git(["rev-parse", "--abbrev-ref", "HEAD"], cwd=repo_root)
    status = _run_git(["status", "--porcelain"], cwd=repo_root)
    return {
        "commit": commit,
        "branch": branch,
        "is_dirty": bool(status),
    }


def _run_git(args: Sequence[str], *, cwd: Path) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=cwd,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip() or None


def _find_git_root(start: Path) -> Path | None:
    current = start.resolve()
    for parent in [current, *current.parents]:
        if (parent / ".git").exists():
            return parent
    return None


def _derive_package_name(run_dir: Path) -> str:
    parts = list(run_dir.parts)
    name_parts: list[str] = []
    if "outputs" in parts:
        idx = parts.index("outputs")
        name_parts = [p for p in parts[idx + 1 :] if p]
    if not name_parts:
        name_parts = [run_dir.name]
    base = "-".join(name_parts)
    return _sanitize_name(f"{base}-model")


def _sanitize_name(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip())
    slug = slug.strip("-_.")
    return slug.lower() if slug else "model-package"


def load_run_config(run_dir: Path) -> DictConfig:
    candidates = [
        run_dir / "checkpoints" / "config.yaml",
        run_dir / ".hydra" / "config.yaml",
        run_dir / "config.yaml",
    ]
    return _load_config_from_candidates(candidates, run_dir)


def discover_training_runs(base_dir: Path) -> list[tuple[Path, DictConfig]]:
    base_path = Path(base_dir).expanduser().resolve()
    if not base_path.exists():
        raise FileNotFoundError(f"Base directory not found: {base_path}")

    runs: list[tuple[Path, DictConfig]] = []
    for config_path in base_path.rglob("checkpoints/config.yaml"):
        run_dir = config_path.parent.parent
        try:
            cfg = load_run_config(run_dir)
        except FileNotFoundError:
            continue
        runs.append((run_dir, cfg))
    return runs


def build_auto_package_name(
    run_dir: Path,
    base_dir: Path,
    cfg: DictConfig,
) -> str:
    try:
        relative = run_dir.relative_to(base_dir)
        path_slug = _sanitize_name(relative.as_posix().replace("/", "-"))
    except ValueError:
        path_slug = _sanitize_name(run_dir.as_posix().replace("/", "-"))

    instance_value = OmegaConf.select(cfg, "env.instance")
    instance_slug = _sanitize_name(str(instance_value)) if instance_value else "unknown"

    config_dict = _cfg_to_dict(cfg)
    config_hash = JSSPSolverBase._compute_config_hash(config_dict)

    return _sanitize_name(f"{path_slug}-{instance_slug}-{config_hash}")


def _cfg_to_dict(cfg: DictConfig) -> dict:
    try:
        container = OmegaConf.to_container(cfg, resolve=False)
    except Exception:
        container = None
    if isinstance(container, dict):
        return container
    return {}


def _load_config_from_package(package_root: Path) -> DictConfig:
    candidates = [
        package_root / "artifacts" / "checkpoints" / "config.yaml",
        package_root / "configs" / "hydra" / "config.yaml",
    ]
    return _load_config_from_candidates(candidates, package_root)


def _load_config_from_candidates(
    candidates: Iterable[Path], root_dir: Path
) -> DictConfig:
    for candidate in candidates:
        if candidate.exists():
            cfg = OmegaConf.load(candidate)
            if not isinstance(cfg, DictConfig):
                cfg = OmegaConf.create(cfg)
            return cfg
    raise FileNotFoundError(f"No config.yaml found near {root_dir}")
