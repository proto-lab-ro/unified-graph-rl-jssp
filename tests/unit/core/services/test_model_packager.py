from __future__ import annotations

import json
import tarfile
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from jssp_core.solver.base import JSSPSolverBase
from jssp_gnn.solver import GnnMatrixSolver
from tools.model_packager import (
    ModelPackagingError,
    build_auto_package_name,
    load_run_config,
    package_model_run,
)


def _create_fake_run(tmp_path: Path) -> Path:
    run_dir = tmp_path / "outputs" / "2025-11-18" / "17-19-01"
    checkpoints = run_dir / "checkpoints"
    hydra_dir = run_dir / ".hydra"
    logs_dir = run_dir / "logs"

    checkpoints.mkdir(parents=True, exist_ok=True)
    hydra_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    (checkpoints / "policy_module_final.pt").write_bytes(b"\x01" * 32)
    (checkpoints / "best_model_metadata.txt").write_text("metric: 1.23\n")
    (checkpoints / "config.yaml").write_text(
        "env:\n"
        "  reward_function: demo\n"
        "  observation_provider: demo\n"
        "  instance: ft06\n"
    )
    (hydra_dir / "config.yaml").write_text("foo: bar\n")
    (run_dir / "train.log").write_text("training log")
    (logs_dir / "events.tfevents").write_text("tensorboard")

    return run_dir


def test_package_model_run_creates_archive_with_manifest(tmp_path: Path) -> None:
    run_dir = _create_fake_run(tmp_path)
    storage_dir = tmp_path / "store"
    extra_file = tmp_path / "notes.txt"
    extra_file.write_text("share me")

    result = package_model_run(
        run_dir,
        storage_dir=storage_dir,
        package_name="demo_package",
        include_logs=True,
        extra_files=[extra_file],
        notes="unit test",
    )

    assert result.archive_path.exists()
    assert result.manifest_path.exists()
    assert result.manifest["package_name"] == "demo_package"
    assert result.manifest["git"]["commit"] is None
    assert result.manifest.get("notes") == "unit test"
    assert result.file_count >= 4
    assert result.total_bytes > 0

    with tarfile.open(result.archive_path, "r:gz") as tar:
        members = {Path(member.name) for member in tar.getmembers()}
    expected_files = {
        Path("demo_package/artifacts/checkpoints/policy_module_final.pt"),
        Path("demo_package/artifacts/checkpoints/best_model_metadata.txt"),
        Path("demo_package/logs/train.log"),
        Path("demo_package/logs/tensorboard/events.tfevents"),
        Path("demo_package/extras/notes.txt"),
    }
    assert expected_files.issubset(members)

    manifest_data = json.loads(result.manifest_path.read_text())
    artifact_paths = {item["path"] for item in manifest_data["artifacts"]}
    assert "artifacts/checkpoints/policy_module_final.pt" in artifact_paths
    assert "extras/notes.txt" in artifact_paths
    assert manifest_data["config"]["env"]["reward_function"] == "demo"


def test_package_model_run_requires_checkpoint(tmp_path: Path) -> None:
    run_dir = tmp_path / "outputs" / "2025-01-01" / "00-00-00"
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    with pytest.raises(ModelPackagingError):
        package_model_run(run_dir)


def test_gnn_matrix_solver_from_package(tmp_path: Path) -> None:
    run_dir = _create_fake_run(tmp_path)
    storage_dir = tmp_path / "store"
    result = package_model_run(run_dir, storage_dir=storage_dir, package_name="demo")

    solver = GnnMatrixSolver.from_package(result.archive_path, device="cpu")

    assert solver.package_root is not None
    assert solver.package_manifest is not None
    assert solver.package_manifest["package_name"] == "demo"
    assert solver.cfg is not None
    assert solver.model_path.endswith(".pt")


def test_build_auto_package_name_includes_instance_and_hash(tmp_path: Path) -> None:
    run_dir = _create_fake_run(tmp_path)
    base_dir = tmp_path / "outputs"
    cfg = load_run_config(run_dir)
    name = build_auto_package_name(run_dir, base_dir, cfg)
    config_dict = OmegaConf.to_container(cfg, resolve=True)
    expected_hash = JSSPSolverBase._compute_config_hash(config_dict)
    assert "ft06" in name
    assert expected_hash in name
