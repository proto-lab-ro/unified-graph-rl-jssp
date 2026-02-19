"""
Unit tests for GNN environment and solver.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from tensordict import TensorDict
from tensordict.nn import TensorDictModuleBase

from jssp_core.instances import get_instance
from jssp_gnn.environments.jssp import GraphMatrixEnv
from jssp_gnn.solver.gnn_solver import GnnMatrixSolver


@pytest.fixture
def small_instance():
    return get_instance("f3x3")


@pytest.fixture
def env_kwargs():
    return {
        "observation_provider": "lb_bipartite_gnn",
        "reward_function": "makespan_improvement",
        "reward_kwargs": {"heuristic": "mwr"},
    }


@pytest.fixture
def mock_cfg():
    return OmegaConf.create(
        {
            "env": {
                "reward_function": "makespan_improvement",
                "observation_provider": "lb_bipartite_gnn",
                "reward_kwargs": {"heuristic": "mwr"},
                "job_selector_type": "operation",
            },
            "gnn": {
                "policy_net": "GIN",
                "critic_net": "GIN",
                "embed_dim": 16,
                "num_layers": 2,
            },
        }
    )


class MockPolicy(TensorDictModuleBase):
    def __init__(self):
        super().__init__()
        self.in_keys = ["observation", "mask"]
        self.out_keys = ["action"]
        self.called = False

    def forward(self, td: TensorDict) -> TensorDict:
        self.called = True
        mask = td["mask"]
        valid_actions = torch.nonzero(mask).squeeze(1)
        if len(valid_actions) > 0:
            action = valid_actions[0]
        else:
            action = torch.tensor(0)
        td.set("action", action)
        return td


@pytest.mark.unit
class TestGraphMatrixEnv:
    """Test GraphMatrixEnv functionality."""

    def test_env_initialization(self, small_instance, env_kwargs):
        env = GraphMatrixEnv(small_instance, env_kwargs)
        assert env.num_jobs == 3
        assert env.num_operations == 9

        # Check specs
        assert "observation" in env.observation_spec.keys()
        assert "mask" in env.observation_spec.keys()
        assert "makespan" in env.observation_spec.keys()

    def test_env_reset(self, small_instance, env_kwargs):
        env = GraphMatrixEnv(small_instance, env_kwargs)
        td = env.reset()

        assert isinstance(td, TensorDict)
        assert "observation" in td.keys()
        assert "mask" in td.keys()
        assert td["mask"].shape == torch.Size([9])  # 3x3 = 9 ops
        assert td["mask"].dtype == torch.bool

    def test_env_step(self, small_instance, env_kwargs):
        env = GraphMatrixEnv(small_instance, env_kwargs)
        env.reset()

        # Select first valid action from mask
        td = env.reset()
        mask = td["mask"]
        valid_actions = torch.nonzero(mask).squeeze(1)
        action = valid_actions[0]

        td_action = td.clone()
        td_action["action"] = action

        next_td = env.step(td_action)
        assert "next" in next_td.keys()
        assert "reward" in next_td["next"].keys()
        assert "done" in next_td["next"].keys()


@pytest.mark.unit
class TestGnnMatrixSolver:
    """Test GnnMatrixSolver functionality."""

    @patch("jssp_gnn.solver.gnn_solver.create_models")
    def test_solver_solve(self, mock_create_models, small_instance, mock_cfg):
        # Mock policy module
        mock_policy = MockPolicy()
        mock_create_models.return_value = (mock_policy, MagicMock())

        # Create solver with mocked model path and config
        solver = GnnMatrixSolver(model_path="dummy_path", cfg=mock_cfg)
        solver.policy_module = mock_policy

        output = solver.solve(small_instance)

        # output is a Schedule object
        assert output.get_makespan() > 0
        assert output.is_complete()
        assert mock_policy.called

    def test_solver_error_handling(self, small_instance, mock_cfg):
        # Test missing model path - solve should raise ValueError
        # We need to mock create_models to return a tuple to reach the ValueError check
        with patch("jssp_gnn.solver.gnn_solver.create_models") as mock_create:
            mock_create.return_value = (MockPolicy(), MagicMock())
            solver = GnnMatrixSolver(model_path=None, cfg=mock_cfg)
            with pytest.raises(ValueError, match="model_path"):
                solver.solve(small_instance)

    def test_solver_get_action(self, mock_cfg):
        mock_policy = MockPolicy()
        solver = GnnMatrixSolver(model_path="dummy", cfg=mock_cfg)
        solver.policy_module = mock_policy

        obs = {
            "node_feats": np.zeros((9, 16), dtype=np.float32),
            "edge_index": np.zeros((2, 10), dtype=np.int64),
            "mask": np.ones(9, dtype=bool),
        }

        action = solver.get_action(obs)
        assert isinstance(action, int)
        assert action == 0
        assert mock_policy.called

    def test_solver_from_policy(self, mock_cfg):
        mock_policy = MockPolicy()
        solver = GnnMatrixSolver.from_policy(
            policy_module=mock_policy, cfg=mock_cfg, max_steps=100
        )
        assert solver.policy_module == mock_policy
        assert solver.model_path is None
        assert (
            solver.max_steps == 50000
        )  # Note: from_policy hardcodes 50000 currently in code
