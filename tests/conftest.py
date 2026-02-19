"""
Pytest configuration and fixtures for JSSP GNN tests.
"""

import numpy as np
import pytest
import torch

from jssp_core.domain.domains import JobSelectorType
from jssp_core.environments.jssp import JSSPEnv
from jssp_core.instances import (
    F3X3_INSTANCE,
    FT06_INSTANCE,
    _parse_instance,
)
from jssp_core.schedule import Schedule
from jssp_gnn.utils import get_device


@pytest.fixture(scope="session")
def device():
    """Get the appropriate device for testing."""
    return get_device()


@pytest.fixture
def ft06_instance():
    """Provide FT06 test instance."""
    return _parse_instance(FT06_INSTANCE)


@pytest.fixture
def small_3x3_instance():
    """Provide small 3x3 test instance."""
    return _parse_instance(F3X3_INSTANCE)


@pytest.fixture
def jssp_env_ft06(ft06_instance):
    """Create JSSP environment with FT06 instance."""
    return JSSPEnv(ft06_instance, random_instance=False)


@pytest.fixture
def jssp_env_3x3(small_3x3_instance):
    """Create JSSP environment with 3x3 instance."""
    return JSSPEnv(
        small_3x3_instance,
        random_instance=False,
        job_selector_type=JobSelectorType.JOB,
    )


@pytest.fixture
def schedule_ft06(ft06_instance):
    """Create schedule with FT06 instance."""
    return Schedule(ft06_instance)


@pytest.fixture
def schedule_3x3(small_3x3_instance):
    """Create schedule with 3x3 instance."""
    return Schedule(small_3x3_instance)


@pytest.fixture
def temp_model_path(tmp_path):
    """Provide temporary path for saving/loading models."""
    return tmp_path / "test_model.pt"


@pytest.fixture
def sample_config():
    """Provide sample configuration for testing (legacy format)."""
    return {
        "env": {
            "instance": "jssp_instances/ft10",
            "random_instance": False,
            "reward_function": "dense_shaped",
            "reward_kwargs": {
                "completion_bonus": 1.0,
                "heuristic": "CR",
                "offset": 0.0,
            },
            "max_episode_steps": 500,
        },
        "training": {
            "frames_per_batch": 36,
            "total_frames": 150_000,
            "sub_batch_size": 36,
            "num_epochs": 10,
            "clip_epsilon": 0.4,
            "gamma": 0.99,
            "lmbda": 0.95,
            "entropy_eps": 0.5,
            "max_steps": 500,
            "critic_coef": 0.2,
            "average_gae": False,
            "eval_freq": 100,
            "num_cells": 256,
            "lr": 2e-5,
            "max_grad_norm": 0.5,
            "split_trajs": False,
            "loss_critic_type": "smooth_l1",
            "action_masking": True,
        },
    }


@pytest.fixture
def hydra_gnn_config():
    """Provide Hydra-compatible GNN configuration for testing."""
    from omegaconf import OmegaConf

    config_dict = {
        "env": {
            "instance": "jssp_instances/ft06",
            "random_instance": False,
            "reward_function": "dense_shaped",
            "reward_kwargs": {
                "completion_bonus": 1.0,
                "heuristic": "CR",
                "offset": 0.0,
            },
            "max_episode_steps": 200,
        },
        "training": {
            "frames_per_batch": 50,
            "total_frames": 1000,
            "sub_batch_size": 50,
            "num_epochs": 2,
            "clip_epsilon": 0.4,
            "gamma": 0.99,
            "lmbda": 0.95,
            "entropy_eps": 0.5,
            "max_steps": 200,
            "critic_coef": 0.2,
            "average_gae": False,
            "eval_freq": 500,
            "num_cells": 64,
            "lr": 1e-4,
            "max_grad_norm": 0.5,
            "split_trajs": False,
            "loss_critic_type": "smooth_l1",
            "action_masking": True,
        },
        "log_dir": "logs/test",
        "save_dir": "models/test",
        "project_name": "jssp_gnn_test",
    }

    return OmegaConf.create(config_dict)


@pytest.fixture
def hydra_sb3_config():
    """Provide Hydra-compatible SB3 configuration for testing."""
    from omegaconf import OmegaConf

    config_dict = {
        "env": {
            "instance": "jssp_instances/ft06",
            "random_instance": False,
            "reward_function": "dense_makespan",
            "reward_kwargs": {
                "truncate_if_invalid": False,
                "completion_bonus": 1.0,
            },
            "max_episode_steps": 200,
        },
        "training": {
            "total_timesteps": 1000,
            "learning_rate": 1e-4,
            "n_steps": 64,
            "batch_size": 32,
            "n_epochs": 2,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "ent_coef": 0.01,
            "vf_coef": 0.5,
            "max_grad_norm": 0.5,
            "seed": 42,
        },
        "parallel": {
            "n_envs": 1,
            "vec_env_cls": "dummy",
        },
        "evaluation": {
            "eval_freq": 500,
            "n_eval_episodes": 3,
            "deterministic": True,
        },
        "logging": {
            "use_tensorboard": True,
            "log_interval": 100,
        },
        "save_dir": "models/test_sb3",
        "log_dir": "logs/test_sb3",
    }

    return OmegaConf.create(config_dict)


@pytest.fixture
def create_temp_hydra_config():
    """Factory fixture for creating temporary Hydra configurations."""

    def _create_config(config_dict, temp_dir=None):
        """Create a temporary Hydra configuration structure."""
        import os
        import tempfile

        from omegaconf import OmegaConf

        if temp_dir is None:
            temp_dir = tempfile.mkdtemp()

        # Create config directory
        config_dir = os.path.join(temp_dir, "conf")
        os.makedirs(config_dir, exist_ok=True)

        # Save main config
        main_config_path = os.path.join(config_dir, "config.yaml")
        with open(main_config_path, "w") as f:
            OmegaConf.save(config_dict, f)

        return config_dir, main_config_path

    return _create_config


@pytest.fixture(autouse=True)
def set_random_seed():
    """Set random seeds for reproducible tests."""
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)


@pytest.fixture
def mock_tensorboard_writer():
    """Mock tensorboard writer for testing."""

    class MockWriter:
        def __init__(self):
            self.scalars = {}

        def add_scalar(self, tag, scalar_value, global_step):
            if tag not in self.scalars:
                self.scalars[tag] = []
            self.scalars[tag].append((scalar_value, global_step))

        def close(self):
            pass

    return MockWriter()


# Skip markers for conditional tests
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "gpu: mark test as requiring GPU")
    config.addinivalue_line("markers", "slow: mark test as slow running")


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add skip conditions."""
    skip_gpu = pytest.mark.skip(reason="GPU not available")

    for item in items:
        if "gpu" in item.keywords and not torch.cuda.is_available():
            item.add_marker(skip_gpu)
