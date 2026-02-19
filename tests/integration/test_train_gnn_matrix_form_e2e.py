from __future__ import annotations

import os
from copy import deepcopy
from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

import jssp_core.utils.utils
from jssp_gnn import train_gnn_matrix_form as trainer
from jssp_gnn import utils as gnn_utils
from jssp_gnn.dispatcher import utils_matrix


CONFIG_DIR = Path(__file__).resolve().parents[2] / "conf" / "gnn"


def _compose_training_cfg(tmp_path: Path, env_config: str, extra_overrides: list[str]):
    """Compose a minimal Hydra config for a fast training smoke test."""
    overrides = [
        f"env={env_config}",
        # keep runs tiny and deterministic
        "device=cpu",
        "env.random_instance=false",
        "env.max_episode_steps=30",
        "training.num_epochs=1",
        "training.frames_per_batch=4",
        "training.total_frames=8",
        "training.sub_batch_size=2",
        "training.eval_freq=0",
        "training.max_steps=30",
        "evaluation.type=none",
        "gnn_feature_extractor.hidden_dim=16",
        "gnn_feature_extractor.k_layers=1",
        f"log_dir={tmp_path / 'logs'}",
        f"save_dir={tmp_path / 'checkpoints'}",
        f"hydra.run.dir={tmp_path / 'hydra'}",
        "hydra.output_subdir=null",
        "hydra/job_logging=disabled",
        "hydra/hydra_logging=disabled",
    ] + list(extra_overrides)

    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        cfg = compose(config_name="base_training", overrides=overrides)
    return cfg


def _force_cpu_devices(tmp_dir: Path | None = None):
    """Ensure all training helpers stay on CPU even when CUDA is available."""
    cpu = torch.device("cpu")
    trainer.device = cpu
    utils_matrix.device = cpu
    gnn_utils.device = cpu
    if tmp_dir is not None:
        tmp_dir.mkdir(parents=True, exist_ok=True)
        os.environ["TMPDIR"] = str(tmp_dir)
        os.environ["MP_TMPDIR"] = str(tmp_dir)
    from torchrl.data.replay_buffers import storages as rb_storages

    gnn_utils.LazyTensorStorage = rb_storages.ListStorage
    try:
        torch.multiprocessing.set_sharing_strategy("file_system")
    except RuntimeError:
        # Fallback for platforms that do not support altering the strategy
        pass
    _disable_shared_memory_collectors()


def _disable_shared_memory_collectors():
    """Patch torchrl collectors to avoid shared memory requirements in tests."""
    from torchrl.collectors import collectors as collectors_mod
    from torchrl.data.replay_buffers import replay_buffers as rb_mod

    class _LocalTrajectoryPool:
        def __init__(self, ctx=None, lock: bool = False):
            self._traj_id = torch.zeros((), device="cpu", dtype=torch.int)

        def get_traj_and_increment(self, n=1, device=None):
            start = int(self._traj_id.item())
            out = torch.arange(start, start + n, device=device)
            self._traj_id.fill_(start + n)
            return out

    def _map_weight_no_shared(weight, policy_device):
        is_param = isinstance(weight, torch.nn.Parameter)
        weight = weight.data
        if weight.device != policy_device:
            weight = weight.to(policy_device)
        if is_param:
            return torch.nn.Parameter(weight, requires_grad=False)
        return weight

    class _TestReplayBuffer:
        def __init__(self, *args, batch_size=None, **kwargs):
            self._data: list = []

        def extend(self, data):
            self._data.append(data)

        def __iter__(self):
            items, self._data = self._data, []
            return iter(items)

        def __len__(self):
            return len(self._data)

    collectors_mod._TrajectoryPool = _LocalTrajectoryPool
    collectors_mod._map_weight = _map_weight_no_shared
    rb_mod.ReplayBuffer = _TestReplayBuffer
    gnn_utils.ReplayBuffer = _TestReplayBuffer


def _run_training(cfg, tmp_path: Path):
    _force_cpu_devices(tmp_path / "tmp")

    log_dir, save_dir = jssp_core.utils.utils.setup_directories(cfg)
    logger = trainer.TrainingLogger(log_dir)
    checkpoint_logger = trainer.ModelCheckpointLogger(
        save_dir, logger, keep_all_checkpoints=True
    )

    env = trainer.create_environment(cfg)
    policy_module, value_module = trainer.create_models(env, cfg)

    eval_cfg = deepcopy(cfg)
    eval_cfg.env.random_instance = False
    eval_env = trainer.create_environment(eval_cfg)
    evaluator = trainer.create_evaluator(eval_env, logger, cfg)

    trained_policy = trainer.train_model(
        policy_module=policy_module,
        value_module=value_module,
        env=env,
        cfg=cfg,
        logger=logger,
        checkpoint_logger=checkpoint_logger,
        evaluator=evaluator,
    )
    return trained_policy, Path(save_dir) / "policy_module_final.pt"


@pytest.mark.integration
@pytest.mark.parametrize(
    ("env_config", "extra_overrides"),
    [
        ("gnn_ft06", []),
        ("gnn_ft20", ["training.frames_per_batch=2", "training.total_frames=4"]),
    ],
)
def test_training_starts_for_multiple_gnn_configs(
    tmp_path, env_config, extra_overrides
):
    cfg = _compose_training_cfg(tmp_path, env_config, extra_overrides)

    trained_policy, checkpoint_path = _run_training(cfg, tmp_path)

    assert isinstance(trained_policy, torch.nn.Module)
    assert checkpoint_path.exists()
    assert checkpoint_path.stat().st_size > 0
