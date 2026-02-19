import os
from pathlib import Path

import pytest
import torch

from jssp_core.utils.config_utils import get_new_log_dir
from jssp_gnn.logger import ModelCheckpointLogger, TrainingLogger
from jssp_gnn.utils import concat_node_graph_features


def test_get_new_log_dir_increments(tmp_path):
    base_dir = tmp_path / "runs"
    save_dir = tmp_path / "models"
    prefix = "experiment_"
    os.makedirs(base_dir / f"{prefix}0")

    new_log_dir, new_save_dir = get_new_log_dir(
        base_dir=str(base_dir),
        save_dir=str(save_dir),
        prefix=prefix,
    )

    assert Path(new_log_dir) == base_dir / f"{prefix}1"
    assert Path(new_save_dir) == save_dir / f"{prefix}1"


def test_concat_node_graph_features_handles_batched_and_unbatched():
    node_features = torch.arange(6, dtype=torch.float32).reshape(3, 2)
    graph_emb = torch.tensor([1.0, 2.0])
    combined = concat_node_graph_features(node_features, graph_emb)
    assert combined.shape == (3, 4)
    torch.testing.assert_close(combined[:, :2], node_features)
    torch.testing.assert_close(combined[:, 2:], graph_emb.expand(3, -1))

    batched_nodes = node_features.unsqueeze(0).expand(2, -1, -1)
    batched_graph_emb = torch.stack([graph_emb, graph_emb + 1])
    batched_combined = concat_node_graph_features(batched_nodes, batched_graph_emb)
    assert batched_combined.shape == (2, 3, 4)
    torch.testing.assert_close(batched_combined[:, :, :2], batched_nodes)
    torch.testing.assert_close(
        batched_combined[:, :, 2:], batched_graph_emb.unsqueeze(1).expand(-1, 3, -1)
    )


def test_concat_node_graph_features_raises_for_mismatched_batch_dim():
    node_features = torch.zeros(3, 2)
    batched_graph_emb = torch.zeros(2, 2)
    with pytest.raises(ValueError):
        concat_node_graph_features(node_features, batched_graph_emb)


def test_model_checkpoint_logger_tracks_best_and_history(tmp_path):
    model = torch.nn.Linear(4, 2)
    with TrainingLogger(str(tmp_path / "tb")) as train_logger:
        checkpoint_logger = ModelCheckpointLogger(
            save_dir=str(tmp_path),
            logger=train_logger,
            keep_all_checkpoints=True,
        )

        # First call should save best + history checkpoint
        assert checkpoint_logger.save_checkpoint(model, metric_value=0.5, step=10)
        assert checkpoint_logger.best_metric == pytest.approx(0.5)
        assert Path(tmp_path / "policy_module.pt").exists()
        assert Path(tmp_path / "policy_module_step_10.pt").exists()

        # Worse metric should not update best but still keep history
        assert checkpoint_logger.save_checkpoint(model, metric_value=0.1, step=20)
        assert checkpoint_logger.best_metric == pytest.approx(0.5)
        assert Path(tmp_path / "policy_module_step_20.pt").exists()

        # Better metric updates best and adds history checkpoint
        assert checkpoint_logger.save_checkpoint(model, metric_value=0.7, step=30)
        assert checkpoint_logger.best_metric == pytest.approx(0.7)
        assert Path(tmp_path / "policy_module_step_30.pt").exists()

        assert checkpoint_logger.checkpoint_count == 2
        assert checkpoint_logger.history_checkpoint_count == 3
