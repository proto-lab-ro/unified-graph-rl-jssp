# ASSUMPTION - all graphs in batch have same number of nodes

import signal
import sys
from copy import deepcopy

import hydra
from omegaconf import DictConfig, open_dict

from jssp_core import set_seed
from jssp_core.utils.utils import (
    log_training_config,
    setup_directories,
)
from jssp_gnn.dispatcher.utils_matrix import create_environment, create_models
from jssp_gnn.evaluator import create_evaluator
from jssp_gnn.logger import ModelCheckpointLogger, TrainingLogger
from jssp_gnn.training.loop import train_model
from jssp_gnn.utils import set_device


@hydra.main(version_base=None, config_path="../../conf/gnn", config_name="")
def main(cfg: DictConfig) -> None:
    """Main training function using Hydra for configuration management."""

    # Set seed for reproducibility
    set_seed(cfg.get("seed", 42))

    # Set device automatically if not specified
    set_device(cfg.device)

    # Setup directories for logging and checkpoints
    log_dir, save_dir = setup_directories(cfg)

    # Log configuration and directories to console
    log_training_config(cfg)

    # Create logger and checkpoint manager
    logger = TrainingLogger(log_dir)
    checkpoint_logger = ModelCheckpointLogger(
        save_dir,
        logger,
        keep_all_checkpoints=True,
    )

    # Create environment and models
    env = create_environment(cfg)
    policy_module, value_module = create_models(env, cfg)

    # Create evaluator
    eval_cfg = deepcopy(cfg)
    with open_dict(eval_cfg):
        eval_cfg.env.random_instance = False
        eval_cfg.env.num_envs = 1

    eval_env = create_environment(eval_cfg)
    evaluator = create_evaluator(eval_env, logger, cfg)

    if cfg.get("dry_run", False):
        print(f"Dry run successful for config: {cfg.get('experiment_name', 'unknown')}")
        return

    # Train the model
    def signal_handler(signum, frame):
        print(
            f"\nReceived signal {signum}. Training interrupted. Saving final checkpoint..."
        )
        checkpoint_logger.save_final_checkpoint(policy_module, "policy_module_final.pt")
        logger.close()
        sys.exit(0)

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    trained_policy = train_model(
        policy_module, value_module, env, cfg, logger, checkpoint_logger, evaluator
    )

    return trained_policy


if __name__ == "__main__":
    main()
