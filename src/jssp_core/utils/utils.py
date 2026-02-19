import os

import hydra
from omegaconf import DictConfig, OmegaConf

from jssp_gnn.utils import get_device


def log_training_config(cfg: DictConfig):
    """Log all configuration parameters and directories to console."""
    print("=" * 80)
    print("TRAINING CONFIGURATION")
    print("=" * 80)
    print(f"Log Directory: {cfg.log_dir}")
    print(f"Save Directory: {cfg.save_dir}")
    print(f"Device: {get_device()}")
    print(
        f"Hydra Output Directory: {hydra.core.hydra_config.HydraConfig.get().runtime.output_dir}"
    )
    print("-" * 80)

    def print_dict(d, indent=0):
        for key, value in d.items():
            if isinstance(value, (dict, DictConfig)):
                print("  " * indent + f"{key}:")
                print_dict(value, indent + 1)
            else:
                print("  " * indent + f"{key}: {value}")

    print("Configuration Parameters:")
    print_dict(OmegaConf.to_container(cfg, resolve=True))
    print("=" * 80)
    print("Starting training...")
    print("=" * 80)


def setup_directories(cfg: DictConfig):
    """Set up logging and checkpoint directories."""
    os.makedirs(cfg.log_dir, exist_ok=True)
    os.makedirs(cfg.save_dir, exist_ok=True)

    # Save the complete configuration to the output directory
    config_path = os.path.join(cfg.save_dir, "config.yaml")
    with open(config_path, "w") as f:
        OmegaConf.save(cfg, f)

    return cfg.log_dir, cfg.save_dir
