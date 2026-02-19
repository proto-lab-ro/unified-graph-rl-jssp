"""
Unit tests for GNN training setup utilities.
"""

import os
from unittest.mock import MagicMock, patch

import pytest
import torch
from omegaconf import OmegaConf

from jssp_core.utils.utils import setup_directories
from jssp_gnn.utils.setup import setup_ppo_training_components


@pytest.fixture
def mock_cfg():
    """Create a mock Hydra configuration."""
    return OmegaConf.create(
        {
            "log_dir": "test_logs",
            "save_dir": "test_save",
            "training": {
                "frames_per_batch": 100,
                "total_frames": 1000,
                "split_trajs": False,
                "sub_batch_size": 10,
                "gamma": 0.99,
                "lmbda": 0.95,
                "average_gae": True,
                "clip_epsilon": 0.2,
                "entropy_eps": 0.01,
                "critic_coef": 0.5,
                "loss_critic_type": "l2",
                "normalize_advantage": True,
                "lr": 1e-4,
                "use_lr_scheduler": True,
                "lr_scheduler_type": "linear",
                "num_epochs": 1,
            },
            "env": {"reward_function": "demo", "observation_provider": "demo"},
        }
    )


@pytest.mark.unit
def test_setup_directories(mock_cfg, tmp_path):
    """Test directory setup and config persistence."""
    # Override directories to use tmp_path
    mock_cfg.log_dir = str(tmp_path / "logs")
    mock_cfg.save_dir = str(tmp_path / "save")

    log_dir, save_dir = setup_directories(mock_cfg)

    assert os.path.exists(log_dir)
    assert os.path.exists(save_dir)
    assert os.path.exists(os.path.join(save_dir, "config.yaml"))

    # Verify saved config
    saved_cfg = OmegaConf.load(os.path.join(save_dir, "config.yaml"))
    assert saved_cfg.training.lr == 1e-4


@pytest.mark.unit
def test_setup_ppo_training_components(mock_cfg):
    """Test creation of PPO training components with mocked modules."""
    mock_policy = MagicMock()
    # Mock policy params for optimizer construction - must be real Parameters
    mock_policy.parameters.return_value = [torch.nn.Parameter(torch.zeros(1))]
    mock_value = MagicMock()
    mock_value.parameters.return_value = [torch.nn.Parameter(torch.zeros(1))]
    mock_env = MagicMock()

    # Set specs for env to avoid collector initialization errors if it checks them
    mock_env.action_spec = MagicMock()
    mock_env.observation_spec = MagicMock()
    mock_env.reward_spec = MagicMock()

    with (
        patch("jssp_gnn.utils.setup.SyncDataCollector"),
        patch("jssp_gnn.utils.setup.ReplayBuffer"),
        patch("jssp_gnn.utils.setup.GAE"),
        patch("jssp_gnn.utils.setup.ClipPPOLoss") as mock_loss_cls,
    ):
        # Mock parameters() for ClipPPOLoss instance - must be real Parameters
        mock_loss_cls.return_value.parameters.return_value = [
            torch.nn.Parameter(torch.zeros(1))
        ]

        components = setup_ppo_training_components(
            mock_policy, mock_value, mock_env, mock_cfg
        )

        # components = (collector, advantage_module, replay_buffer, loss_module, optimizer, scheduler)
        assert len(components) == 6
        assert components[4] is not None  # Optimizer
        assert components[5] is not None  # Scheduler (enabled in mock_cfg)


@pytest.mark.unit
@pytest.mark.parametrize(
    "scheduler_type",
    ["linear", "cosine", "exponential", "step", "multistep", "plateau"],
)
def test_all_scheduler_types(mock_cfg, scheduler_type):
    """Verify all supported scheduler types can be instantiated."""
    mock_cfg.training.lr_scheduler_type = scheduler_type

    mock_policy = MagicMock()
    mock_policy.parameters.return_value = [torch.nn.Parameter(torch.zeros(1))]
    mock_value = MagicMock()
    mock_value.parameters.return_value = [torch.nn.Parameter(torch.zeros(1))]
    mock_env = MagicMock()

    with (
        patch("jssp_gnn.utils.setup.SyncDataCollector"),
        patch("jssp_gnn.utils.setup.ReplayBuffer"),
        patch("jssp_gnn.utils.setup.GAE"),
        patch("jssp_gnn.utils.setup.ClipPPOLoss") as mock_loss_cls,
    ):
        mock_loss_cls.return_value.parameters.return_value = [
            torch.nn.Parameter(torch.zeros(1))
        ]

        _, _, _, _, _, scheduler = setup_ppo_training_components(
            mock_policy, mock_value, mock_env, mock_cfg
        )

        assert scheduler is not None
