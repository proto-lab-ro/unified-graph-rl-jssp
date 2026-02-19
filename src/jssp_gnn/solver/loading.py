import hashlib
import json
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from omegaconf import OmegaConf


if TYPE_CHECKING:
    from jssp_gnn.solver import GnnMatrixSolver


def _format_solver_name(
    root_dir: Path,
    checkpoints_dir: Path,
    instance_name: str | None,
    observation_provider: str | None,
    reward_name: str | None,
) -> str:
    """Create a readable solver name based on run folder and instance."""
    run_label_path: Path | None = None
    try:
        run_label_path = checkpoints_dir.relative_to(root_dir)
    except ValueError:
        run_label_path = checkpoints_dir

    if run_label_path.name == "checkpoints":
        run_label_path = run_label_path.parent

    run_label = run_label_path.as_posix().lstrip("./")
    if not run_label:
        run_label = checkpoints_dir.parent.name

    run_label = run_label.replace("/", "_")
    instance_suffix = (Path(instance_name).name if instance_name else "unknown").lower()
    observation_suffix = (
        str(observation_provider).replace("/", "_").lower()
        if observation_provider
        else "unknown"
    )
    reward_suffix = (
        str(reward_name).replace("/", "_").lower() if reward_name else "unknown"
    )
    return f"GNN_{instance_suffix}_{observation_suffix}_{reward_suffix}_{run_label}"


def _compute_config_hash(cfg: Any) -> str:
    try:
        container = OmegaConf.to_container(cfg, resolve=False)
    except Exception:
        container = None

    try:
        if container is not None:
            serialized = json.dumps(
                container,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
            )
        else:
            raise TypeError("Unserializable container")
    except (TypeError, ValueError):
        serialized = OmegaConf.to_yaml(cfg, resolve=False)

    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:12]


def _matches_expected(value: Any, expected: Any) -> bool:
    if value is None:
        return False

    if callable(expected):
        return bool(expected(value))

    if isinstance(expected, str):
        return expected in str(value)

    if isinstance(expected, Iterable) and not isinstance(expected, (str, bytes)):
        return any(_matches_expected(value, item) for item in expected)

    return value == expected


def _config_matches_filters(cfg: Any, config_filters: dict[str, Any]) -> bool:
    for path, expected in config_filters.items():
        value = OmegaConf.select(cfg, path)
        if not _matches_expected(value, expected):
            return False
    return True


def load_gnn_solvers(
    base_dirs: str | Path | Iterable[str | Path],
    *,
    instance_filter: str | None = None,
    config_filters: dict[str, Any] | None = None,
    config_filter: Callable[[Any], bool] | None = None,
    model_filename: str = "policy_module_final.pt",
) -> list["GnnMatrixSolver"]:
    """
    Discover all trained GNN solvers under one or multiple base directories matching the optional filters.

    Args:
        base_dirs: Root directory (or iterable of directories) containing experiment sub-folders.
        instance_filter: (Legacy) substring that must appear in env.instance.
        config_filters: Dict of Hydra-style dot paths -> expected values/substrings/callables.
        config_filter: Callable receiving the loaded config and returning True/False.
        model_filename: Checkpoint filename to load within each run's checkpoint directory.
    """
    if isinstance(base_dirs, (str, Path)):
        normalized_base_dirs = [Path(base_dirs)]
    else:
        normalized_base_dirs = [Path(path) for path in base_dirs]

    if not normalized_base_dirs:
        raise ValueError("At least one base directory must be provided.")

    for base_path in normalized_base_dirs:
        if not base_path.exists():
            raise FileNotFoundError(f"Base directory does not exist: {base_path}")

    filters = dict(config_filters or {})
    if instance_filter and "env.instance" not in filters:
        filters["env.instance"] = instance_filter

    config_entries: list[tuple[Path, Path]] = []
    for base_path in normalized_base_dirs:
        config_paths = {*(base_path.glob("**/checkpoints/config.yaml"))}

        direct_config = base_path / "config.yaml"
        if direct_config.is_file():
            config_paths.add(direct_config)

        for config_path in config_paths:
            config_entries.append((config_path, base_path))

    from jssp_gnn.solver import GnnMatrixSolver

    solvers: list[GnnMatrixSolver] = []

    for config_path, root_path in sorted(
        config_entries, key=lambda item: (str(item[0]), str(item[1]))
    ):
        checkpoints_dir = config_path.parent
        model_path = checkpoints_dir / model_filename
        if not model_path.is_file():
            continue

        cfg = OmegaConf.load(config_path)

        if filters and not _config_matches_filters(cfg, filters):
            continue

        if config_filter and not config_filter(cfg):
            continue

        env_instance = str(OmegaConf.select(cfg, "env.instance") or "")
        solver = GnnMatrixSolver(str(model_path))
        solver.load_config_from_yaml(config_path)
        observation_provider = str(
            OmegaConf.select(cfg, "env.observation_provider") or ""
        )
        reward_function = str(OmegaConf.select(cfg, "env.reward_function") or "")
        config_hash = _compute_config_hash(cfg)
        solver._name = _format_solver_name(
            root_path,
            checkpoints_dir,
            env_instance,
            observation_provider,
            reward_function,
        )
        solver.config_hash = config_hash
        solvers.append(solver)

    return solvers
