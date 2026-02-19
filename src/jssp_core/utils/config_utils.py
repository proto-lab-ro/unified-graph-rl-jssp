import os
from pathlib import Path

import yaml


def load_config(file_path: str | Path = "src/jssp_gnn/config_default.yaml") -> dict:
    """Load configuration from yaml file.

    Args:
        file_path: Path to the configuration file.

    Returns:
        dict: Configuration dictionary
    """
    path = Path(file_path)
    if not path.exists():
        # Fallback to relative to project root if called from elsewhere
        root_path = Path(__file__).parent.parent.parent.parent / file_path
        if root_path.exists():
            path = root_path

    with path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def get_new_log_dir(
    base_dir: str = "./logs/jssp_gnn",
    save_dir: str = "./best_model/",
    prefix: str = "run_",
) -> tuple[str, str]:
    """Create new log and save directories with incremental naming."""
    os.makedirs(base_dir, exist_ok=True)
    i = 0
    while os.path.exists(os.path.join(base_dir, f"{prefix}{i}")):
        i += 1
    return os.path.join(base_dir, f"{prefix}{i}"), os.path.join(
        save_dir, f"{prefix}{i}/"
    )


def setup_dirs(config: dict, hpo: bool = False) -> tuple[str, str]:
    """Set up logging and save directories based on config.

    Args:
        config: Configuration dictionary
        hpo: Whether this is for hyperparameter optimization

    Returns:
        tuple: (log_dir, save_dir) paths
    """

    log_dir, save_dir = get_new_log_dir()

    return log_dir, save_dir


def save_config(config: dict, save_dir: str | Path) -> None:
    """Save configuration to specified directory.

    Args:
        config: Configuration dictionary
        save_dir: Directory to save the configuration
    """
    path = Path(save_dir)
    print(f"Saving config to {path}")
    path.mkdir(parents=True, exist_ok=True)
    with (path / "config_default.yaml").open("w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
