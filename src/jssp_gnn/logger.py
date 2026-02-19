"""
Logging utilities for JSSP GNN training.

This module provides a reusable logging class for training metrics,
evaluation results, and other relevant information during the training process.
"""

from collections.abc import Callable
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter


class TrainingLogger:
    """
    A wrapper class for logging training and evaluation metrics.

    This class abstracts TensorBoard logging functionality and provides
    convenient methods for logging different types of metrics during training.
    """

    def __init__(self, log_dir: str):
        """
        Initialize the logger with a TensorBoard writer.

        Args:
            log_dir: Directory path for TensorBoard logs
        """
        self.writer = SummaryWriter(log_dir)
        self.step_count = 0

    def log_model_metrics(
        self,
        actor_weight_dif: float,
        critic_weight_dif: float,
        critic_gnn_weight_dif: float,
        actor_gnn_weight_dif: float,
        step,
    ):
        self.writer.add_scalar("model/actor_weight_dif", actor_weight_dif, step)
        self.writer.add_scalar("model/critic_weight_dif", critic_weight_dif, step)
        self.writer.add_scalar(
            "model/critic_gnn_weight_dif", critic_gnn_weight_dif, step
        )  # actor_gnn_weight_dif
        self.writer.add_scalar(
            "model/actor_gnn_weight_dif", actor_gnn_weight_dif, step
        )  # actor_gnn_weight_dif
        self.writer.add_scalar(
            "model/only_actor_weight_dif", actor_weight_dif - actor_gnn_weight_dif, step
        )
        self.writer.add_scalar(
            "model/only_critic_weight_dif",
            critic_weight_dif - critic_gnn_weight_dif,
            step,
        )

    def log_loss_metrics(
        self,
        loss_value: float,
        kl_approx: float,
        clip_fraction: float,
        loss_objective: float,
        loss_critic: float,
        step: int | None = None,
        agent=None,
        explained_variance: float | None = None,
    ):
        """
        Log training-related metrics.

        Args:
            loss_value: Total loss value
            kl_approx: KL divergence approximation
            clip_fraction: PPO clipping fraction
        """
        step = step or self.step_count
        if agent is not None:
            p = "loss/" + agent + "/"
        else:
            p = "loss/"

        self.writer.add_scalar(p + "loss_value", loss_value, step)
        self.writer.add_scalar(p + "kl_approx", kl_approx, step)
        self.writer.add_scalar(p + "clip_fraction", clip_fraction, step)
        self.writer.add_scalar(p + "loss_objective", loss_objective, step)
        if loss_critic is not None:
            self.writer.add_scalar(p + "loss_critic", loss_critic, step)
        if explained_variance is not None:
            self.writer.add_scalar(p + "explained_variance", explained_variance, step)

    def log_critic_metrics(
        self,
        critic_loss: float,
        step: int | None = None,
    ):
        """
        Log critic-related metrics.

        Args:
            value_loss: Loss from the value function
            value_mean: Mean of the value estimates
            value_std: Standard deviation of the value estimates
            step: Current step/frame count (optional)
        """
        step = step or self.step_count

        p = "loss/critic_loss"

        self.writer.add_scalar(p, critic_loss, step)

    def log_training_lr(
        self,
        learning_rate: float,
        step: int | None = None,
        agent=None,
    ):
        step = step or self.step_count
        p = "train/"
        self.writer.add_scalar(p + "lr", learning_rate, step)

    def log_training_metrics(
        self,
        learning_rate: float,
        mean_reward: float,
        sum_reward: float,
        step: int | None = None,
        agent=None,
        logits_std: float | None = None,
        logits_mean: float | None = None,
        action_masking_rate: float | None = None,
    ):
        """
        Log training-related metrics.

        Args:
            learning_rate: Current learning rate
            mean_reward: Mean reward for the batch
            sum_reward: Sum of rewards for the batch
            step: Current step/frame count (optional)
            logits_std: Standard deviation of logits (optional)
            logits_mean: Mean of logits (optional)
            action_masking_rate: Rate of masked actions (optional)
        """
        step = step or self.step_count

        if agent is not None:
            p = "train/" + agent + "/"
        else:
            p = "train/"
        self.writer.add_scalar(p + "lr", learning_rate, step)
        self.writer.add_scalar(p + "mean_reward", mean_reward, step)
        self.writer.add_scalar(p + "sum_reward", sum_reward, step)

        if logits_std is not None:
            self.writer.add_scalar(p + "logits_std", logits_std, step)
        if logits_mean is not None:
            self.writer.add_scalar(p + "logits_mean", logits_mean, step)
        if action_masking_rate is not None:
            self.writer.add_scalar(p + "action_masking_rate", action_masking_rate, step)

    def log_episode_metrics(
        self,
        max_return: float,
        avg_return: float,
        max_length: int,
        min_makespan: float,
        mean_makespan: float,
        count_terminated: int,
        num_agvs: int = 0,
        step: int | None = None,
        agent=None,
    ):
        """
        Log episode-related metrics.

        Args:
            max_return: Maximum episode return in the batch
            avg_return: Average episode return for completed episodes
            max_length: Maximum episode length
            step: Current step/frame count (optional)
        """
        step = step or self.step_count
        if agent is not None:
            p = "episode/" + agent + "/"
        else:
            p = "episode/"

        self.writer.add_scalar(p + "max_return", max_return, step)
        self.writer.add_scalar(p + "avg_return", avg_return, step)
        self.writer.add_scalar(p + "max_length", max_length, step)
        self.writer.add_scalar(p + "min_makespan", min_makespan, step)
        self.writer.add_scalar(p + "mean_makespan", mean_makespan, step)
        self.writer.add_scalar(p + "count_terminated", count_terminated, step)
        self.writer.add_scalar(p + "num_agvs", num_agvs, step)

    def log_evaluation_metrics(
        self,
        mean_reward: float,
        sum_reward: float,
        step_count: int,
        avg_return: float,
        makespan: float,
        step: int | None = None,
        agent=None,
    ):
        """
        Log evaluation metrics.

        Args:
            mean_reward: Mean reward during evaluation
            sum_reward: Sum of rewards during evaluation
            step_count: Number of steps taken during evaluation
            avg_return: Average return for completed episodes
            makespan: Final makespan from the schedule
            step: Current step/frame count (optional)
        """
        step = step or self.step_count
        if agent is not None:
            p = "eval/" + agent + "/"
        else:
            p = "eval/"
        self.writer.add_scalar(p + "mean_reward", mean_reward, step)
        self.writer.add_scalar(p + "sum_reward", sum_reward, step)
        self.writer.add_scalar(p + "step_count", step_count, step)
        self.writer.add_scalar(p + "avg_return", avg_return, step)
        self.writer.add_scalar(p + "makespan", makespan, step)

    def log_detailed_benchmark_metrics(
        self,
        step: int | None = None,
        all_results: dict | None = None,
    ):
        """
        Log detailed benchmark evaluation metrics.

        Args:
            step: Current step/frame count (optional)
            all_results: Dictionary of detailed results from benchmark evaluation
        """

        step = step or self.step_count
        if all_results is not None:
            for (instance_name, num_agvs), metrics in all_results.items():
                # Extract just the instance name (e.g., "la09" from "jssp_instances/transport/la09")
                short_name = Path(instance_name).name
                for metric_name, value in metrics.items():
                    tag = f"benchmark/{short_name}/{metric_name}"
                    self.writer.add_scalar(tag, value, step)

    def log_benchmark_evaluation_metrics(
        self,
        avg_return: float,
        makespan: float,
        step: int | None = None,
        agent=None,
    ):
        """
        Log evaluation metrics.

        Args:
            mean_reward: Mean reward during evaluation
            sum_reward: Sum of rewards during evaluation
            step_count: Number of steps taken during evaluation
            avg_return: Average return for completed episodes
            makespan: Final makespan from the schedule
            step: Current step/frame count (optional)
        """
        step = step or self.step_count
        if agent is not None:
            p = "eval/" + agent + "/"
        else:
            p = "eval/"
        self.writer.add_scalar(p + "avg_return", avg_return, step)
        self.writer.add_scalar(p + "makespan", makespan, step)

    def log_custom_metrics(self, metrics: dict[str, float], step: int | None = None):
        """
        Log multiple custom metrics at once.

        Args:
            metrics: Dictionary of metric names and values
            step: Current step/frame count (optional)
        """
        step = step or self.step_count
        for tag, value in metrics.items():
            self.writer.add_scalar(tag, value, step)

    def update_step(self, step: int):
        """Update the internal step counter."""
        self.step_count = step

    def close(self):
        """Close the TensorBoard writer."""
        self.writer.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensures writer is closed."""
        self.close()


class ModelCheckpointLogger:
    """
    A utility class for logging model checkpoints and best model tracking.
    """

    def __init__(
        self,
        save_dir: str,
        logger: TrainingLogger,
        keep_all_checkpoints: bool = False,
        clean_dict: Callable | None = None,
    ):
        """
        Initialize the checkpoint logger.

        Args:
            save_dir: Directory to save model checkpoints
            logger: TrainingLogger instance for logging checkpoint events
            keep_all_checkpoints: If True, every checkpoint is saved with a unique
                filename in addition to the best checkpoint file.
        """
        self.save_dir = save_dir
        self.logger = logger
        self.best_metric = -float("inf")
        self.checkpoint_count = 0
        self.keep_all_checkpoints = keep_all_checkpoints
        self.history_checkpoint_count = 0
        self.clean_dict = clean_dict

    def _save_model_state(self, model: torch.nn.Module | tuple, filename: str) -> None:
        """
        Save model (and optional auxiliary modules) to disk.

        Args:
            model: Model or tuple of (aux_models, main_model)
            filename: Target filename relative to save_dir
        """
        save_dir_path = Path(self.save_dir)

        # Determine main model and any auxiliary modules
        if isinstance(model, tuple):
            aux_modules, main_model = model
        else:
            aux_modules, main_model = {}, model

        # Collect tasks: (module, filename, description)
        tasks = [(main_model, filename, "Model checkpoint")]
        for key, module in aux_modules.items():
            tasks.append((module, f"{key}_{filename}", f"Auxiliary model '{key}'"))

        for module, fname, description in tasks:
            target_path = save_dir_path / fname
            state_dict = module.state_dict()
            if self.clean_dict:
                state_dict = self.clean_dict(state_dict)

            torch.save(state_dict, target_path)
            print(f"{description} saved to {target_path}")

    def _build_history_filename(self, filename: str, step: int | None) -> str:
        """
        Construct a unique filename for historical checkpoints.
        """
        base = Path(filename)
        suffix = (
            f"_step_{step}"
            if step is not None
            else f"_ckpt_{self.history_checkpoint_count + 1}"
        )
        return f"{base.stem}{suffix}{base.suffix}"

    def save_checkpoint(
        self,
        model: torch.nn.Module,
        metric_value: float,
        metric_name: str = "eval_reward",
        filename: str = "policy_module.pt",
        step: int | None = None,
    ) -> bool:
        """
        Save model checkpoint and optionally keep every checkpoint.

        Args:
            model: The model to save
            metric_value: Current metric value
            metric_name: Name of the metric being tracked
            filename: Filename for the saved model
            step: Current training step

        Returns:
            bool: True if any checkpoint was saved, False otherwise
        """
        saved_any_checkpoint = False
        is_new_best = metric_value > self.best_metric

        if is_new_best:
            self.best_metric = metric_value
            self._save_model_state(model, filename)

            if step is not None:
                self.logger.log_custom_metrics(
                    {f"checkpoint/{metric_name}_best": metric_value}, step=step
                )

            print(f"New best model saved with {metric_name}: {metric_value:.4f}")
            self.checkpoint_count += 1
            saved_any_checkpoint = True

        if self.keep_all_checkpoints:
            history_filename = self._build_history_filename(filename, step)
            self._save_model_state(model, history_filename)
            self.history_checkpoint_count += 1
            saved_any_checkpoint = True

        return saved_any_checkpoint

    def save_final_checkpoint(
        self, model: torch.nn.Module, filename: str = "policy_module_final.pt"
    ):
        """
        Save final model checkpoint regardless of performance.

        Args:
            model: The model to save
            filename: Filename for the saved model
        """

        self._save_model_state(model, filename)
        print(f"Final model saved to {Path(self.save_dir) / filename}")
