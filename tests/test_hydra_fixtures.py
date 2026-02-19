"""
Test the new Hydra configuration fixtures.
"""

from omegaconf import DictConfig, OmegaConf


def test_hydra_gnn_config_fixture(hydra_gnn_config):
    """Test that the Hydra GNN config fixture works correctly."""
    assert isinstance(hydra_gnn_config, DictConfig)

    # Test structure
    assert "env" in hydra_gnn_config
    assert "training" in hydra_gnn_config

    # Test env settings
    assert hydra_gnn_config.env.instance == "jssp_instances/ft06"
    assert hydra_gnn_config.env.random_instance is False
    assert hydra_gnn_config.env.max_episode_steps == 200

    # Test training settings
    assert hydra_gnn_config.training.frames_per_batch == 50
    assert hydra_gnn_config.training.total_frames == 1000
    assert hydra_gnn_config.training.lr == 1e-4


def test_hydra_sb3_config_fixture(hydra_sb3_config):
    """Test that the Hydra SB3 config fixture works correctly."""
    assert isinstance(hydra_sb3_config, DictConfig)

    # Test structure
    assert "env" in hydra_sb3_config
    assert "training" in hydra_sb3_config
    assert "parallel" in hydra_sb3_config
    assert "evaluation" in hydra_sb3_config

    # Test env settings
    assert hydra_sb3_config.env.instance == "jssp_instances/ft06"
    assert hydra_sb3_config.env.reward_function == "dense_makespan"

    # Test training settings
    assert hydra_sb3_config.training.total_timesteps == 1000
    assert hydra_sb3_config.training.learning_rate == 1e-4

    # Test parallel settings
    assert hydra_sb3_config.parallel.n_envs == 1


def test_create_temp_hydra_config_fixture(create_temp_hydra_config, tmp_path):
    """Test the temporary Hydra config creation fixture."""
    import os

    test_config = OmegaConf.create({"test_param": 42, "nested": {"value": "test"}})

    config_dir, config_path = create_temp_hydra_config(test_config, str(tmp_path))

    # Check that files were created
    assert os.path.exists(config_dir)
    assert os.path.exists(config_path)

    # Test loading the saved config
    loaded_config = OmegaConf.load(config_path)
    assert loaded_config.test_param == 42
    assert loaded_config.nested.value == "test"


def test_config_compatibility_with_omegaconf():
    """Test that our fixtures are compatible with OmegaConf operations."""
    from omegaconf import OmegaConf

    # Test merging configs
    base_config = OmegaConf.create(
        {"env": {"instance": "base"}, "training": {"lr": 1e-3}}
    )

    override_config = OmegaConf.create({"training": {"lr": 1e-4, "epochs": 10}})

    merged = OmegaConf.merge(base_config, override_config)

    assert merged.env.instance == "base"
    assert merged.training.lr == 1e-4
    assert merged.training.epochs == 10
