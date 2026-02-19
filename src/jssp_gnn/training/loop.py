import torch
from omegaconf import DictConfig
from tqdm import tqdm

from jssp_gnn.curriculum import get_curriculum_manager
from jssp_gnn.evaluator import BaseEvaluator
from jssp_gnn.logger import ModelCheckpointLogger, TrainingLogger
from jssp_gnn.utils import get_device
from jssp_gnn.utils.setup import setup_ppo_training_components


def check_stop(
    value: float,
    patience: int,
    mode: str,
    target: float | None,
    state: dict[str, int],
    gap: float = 0.0,
) -> bool:
    """
    Return True if the eval metric meets the target (with optional gap) for
    `patience` consecutive evaluations.

    Args:
        value: Current eval metric value.
        patience: Required consecutive hits to stop.
        mode: "min" (lower is better) or "max" (higher is better).
        target: Target threshold to compare against.
        state: Mutable dict tracking the current hit streak, e.g., {"streak": 0}.
        gap: Additional margin required beyond the target (default 0).
    """
    if target is None:
        return False

    mode = mode.lower()
    if mode == "min":
        hit = value <= (target - gap)
    elif mode == "max":
        hit = value >= (target + gap)
    else:
        raise ValueError(f"Unknown mode '{mode}', expected 'min' or 'max'.")

    state["streak"] = state.get("streak", 0) + 1 if hit else 0
    print(f"Early stopping streak: {state['streak']} / {patience} (hit: {hit})")
    return state["streak"] >= patience


def set_env_masking(env, enabled: bool):
    """
    Enable or disable action masking in the environment.

    Args:
        env: The environment (possibly wrapped)
        enabled: Whether masking should be enabled
    """
    # Try calling directly (handles ParallelEnv and properly proxying wrappers)
    try:
        if hasattr(env, "set_masking_enabled"):
            env.set_masking_enabled(enabled)
            return
        # ParallelEnv might not have the attribute in dir() but supports it via __getattr__
        # so we try calling it anyway if it's not explicitly missing
        # However, purely relying on exception is safer
        env.set_masking_enabled(enabled)
        return
    except (AttributeError, RuntimeError):
        # Fallback for wrappers that don't proxy or other issues
        pass

    # Handle TransformedEnv or other wrappers manually if proxy failed
    current_env = env
    while hasattr(current_env, "base_env"):
        if hasattr(current_env, "set_masking_enabled"):
            current_env.set_masking_enabled(enabled)
            return
        current_env = current_env.base_env

    # Check the final unwrapped environment
    if hasattr(current_env, "set_masking_enabled"):
        current_env.set_masking_enabled(enabled)


def train_model(
    policy_module,
    value_module,
    env,
    cfg: DictConfig,
    logger: TrainingLogger,
    checkpoint_logger: ModelCheckpointLogger,
    evaluator: BaseEvaluator | None = None,
):
    """
    Main training loop.

    Args:
        policy_module: The policy network
        value_module: The value network
        env: The training environment
        cfg: Configuration object
        logger: Training logger
        checkpoint_logger: Checkpoint logger
        evaluator: Optional evaluator for running evaluation during training
    """
    device = get_device()
    (
        collector,
        advantage_module,
        replay_buffer,
        loss_module,
        optimizer,
        scheulder,
    ) = setup_ppo_training_components(policy_module, value_module, env, cfg)

    pbar = tqdm(total=cfg.training.total_frames)
    num_updates = 0
    rollout_frames = 0
    checkpoint_freq = cfg.training.get("checkpoint_freq")
    if checkpoint_freq is not None and checkpoint_freq <= 0:
        print("checkpoint_freq is non-positive; periodic checkpoints are disabled.")
        checkpoint_freq = None
    next_checkpoint_frame = checkpoint_freq

    eval_freq = cfg.training.get("eval_freq")
    if eval_freq is not None and eval_freq <= 0:
        print(f"eval_freq is {eval_freq}; frame-based evaluation is disabled.")
        eval_freq = None

    eval_rollout_freq = cfg.training.get("eval_rollout_freq")
    if eval_rollout_freq is not None and eval_rollout_freq <= 0:
        print(
            f"eval_rollout_freq is {eval_rollout_freq}; rollout-based evaluation is disabled."
        )
        eval_rollout_freq = None

    # Run eval once for initial model
    if evaluator is not None:
        _ = evaluator.evaluate(policy_module, 0)

    # Curriculum setup
    curriculum_manager = get_curriculum_manager(cfg)

    warmup_steps = cfg.training.get("warmup_steps", 0)
    masking_enabled = rollout_frames >= warmup_steps
    set_env_masking(env, masking_enabled)
    if not masking_enabled:
        print(
            f"Starting with action masking DISABLED (warmup for {warmup_steps} steps)"
        )

    early_stop_state = {"streak": 0}
    total_rollouts = 0
    while rollout_frames < cfg.training.total_frames:
        # Check for curriculum update at start of loop (or if we broke out)
        if curriculum_manager.should_update(rollout_frames):
            env, collector, replay_buffer = curriculum_manager.update(
                rollout_frames, env, collector, replay_buffer, policy_module
            )
            # Ensure masking state is preserved across curriculum updates
            set_env_masking(env, masking_enabled)

        for i, tensordict_data in enumerate(collector):
            total_rollouts += 1
            number_frames = tensordict_data.numel()
            rollout_frames += number_frames

            # Update masking based on warmup
            new_masking_enabled = rollout_frames >= warmup_steps
            if new_masking_enabled != masking_enabled:
                masking_enabled = new_masking_enabled
                set_env_masking(env, masking_enabled)
                if masking_enabled:
                    print(
                        f"Step {rollout_frames}: Warmup complete. Action masking ENABLED."
                    )
                else:
                    print(f"Step {rollout_frames}: Action masking DISABLED.")

            # Metrics accumulators for the update cycle
            cycle_losses = {
                "loss_value": 0.0,
                "kl_approx": 0.0,
                "clip_fraction": 0.0,
                "loss_objective": 0.0,
                "loss_critic": 0.0,
                "explained_variance": 0.0,
            }
            cycle_update_count = 0

            for _ in range(cfg.training.num_epochs):
                with torch.no_grad():
                    data = advantage_module(tensordict_data)
                data_reshape = data.reshape(-1)
                replay_buffer.extend(data_reshape)
                for batch in replay_buffer:
                    batch = batch.to(device)
                    loss_vals = loss_module(batch)
                    loss_value = (
                        loss_vals["loss_objective"]
                        + loss_vals["loss_critic"]
                        + loss_vals["loss_entropy"]
                    )

                    # Backward pass and optimization
                    loss_value.backward()
                    torch.nn.utils.clip_grad_norm_(
                        loss_module.parameters(), cfg.training.max_grad_norm
                    )
                    num_updates += 1

                    # Calculate explained variance
                    # data has "state_value" (predicted) and "value_target" (target)
                    # batch is a subset of data (reshuffled), so we use batch
                    y_true = batch.get("value_target")
                    y_pred = batch.get("state_value")

                    explained_var = 0.0
                    if y_true is not None and y_pred is not None:
                        y_var = torch.var(y_true)
                        if y_var > 1e-8:
                            explained_var = 1 - torch.var(y_true - y_pred) / y_var
                            explained_var = explained_var.item()
                        else:
                            explained_var = float("nan")

                    # Accumulate metrics
                    cycle_losses["loss_value"] += loss_value.item()
                    cycle_losses["kl_approx"] += loss_vals.get(
                        "kl_approx", torch.tensor(0.0)
                    ).item()
                    cycle_losses["clip_fraction"] += loss_vals.get(
                        "clip_fraction", torch.tensor(0.0)
                    ).item()
                    cycle_losses["loss_objective"] += loss_vals["loss_objective"].item()
                    cycle_losses["loss_critic"] += loss_vals["loss_critic"].item()
                    cycle_losses["explained_variance"] += (
                        explained_var if explained_var == explained_var else 0.0
                    )  # Handle NaN
                    cycle_update_count += 1

                    optimizer.step()
                    optimizer.zero_grad()
                # step scheduler once per rollout
                if scheulder is not None:
                    if cfg.training.get("lr_scheduler_type") == "plateau":
                        # For ReduceLROnPlateau, step with the metric
                        scheulder.step(tensordict_data["next", "reward"].mean().item())
                    else:
                        scheulder.step()

            # Log averaged loss metrics for the update cycle
            if cycle_update_count > 0:
                logger.log_loss_metrics(
                    loss_value=cycle_losses["loss_value"] / cycle_update_count,
                    kl_approx=cycle_losses["kl_approx"] / cycle_update_count,
                    clip_fraction=cycle_losses["clip_fraction"] / cycle_update_count,
                    loss_objective=cycle_losses["loss_objective"] / cycle_update_count,
                    loss_critic=cycle_losses["loss_critic"] / cycle_update_count,
                    explained_variance=cycle_losses["explained_variance"]
                    / cycle_update_count,
                    step=rollout_frames,
                )

            # all n rollout

            # if i % 1 == 0:
            # Episode and training metrics logging
            mask = tensordict_data["next", "done"]
            terminated = tensordict_data.get(("next", "terminated"))
            mean_reward = tensordict_data["next", "reward"].mean().item()
            sum_reward = tensordict_data["next", "reward"].sum().item()

            # Calculate logit statistics
            logits_std = None
            logits_mean = None
            action_masking_rate = None

            with torch.no_grad():
                logits = tensordict_data.get("logits")
                if logits is None:
                    # Compute logits if not present
                    # policy_module.module is the TensorDictModule that outputs logits
                    td_subset = tensordict_data.select(*policy_module.module.in_keys)
                    policy_module.module(td_subset)
                    logits = td_subset.get("logits")

                if logits is not None:
                    logits_std = logits.std().item()
                    logits_mean = logits.mean().item()

                # Calculate action masking rate
                action_mask = tensordict_data.get("mask")
                if action_mask is None:
                    action_mask = tensordict_data.get(("observation", "mask"))

                if action_mask is not None:
                    # 1.0 means valid, 0.0 means masked
                    # We want % of masked out actions
                    action_masking_rate = 1.0 - action_mask.float().mean().item()

            # Log training metrics
            logger.log_training_metrics(
                learning_rate=optimizer.param_groups[0]["lr"],
                mean_reward=mean_reward,
                sum_reward=sum_reward,
                step=rollout_frames,
                logits_std=logits_std,
                logits_mean=logits_mean,
                action_masking_rate=action_masking_rate,
            )

            # Only log episode metrics if there are completed episodes
            episode_rewards = tensordict_data["next", "episode_reward"][mask]
            makespans = (
                tensordict_data["next", "makespan"][terminated]
                if terminated is not None
                else torch.tensor([])
            )

            if episode_rewards.numel() > 0:
                max_return = episode_rewards.max().item()
                avg_return = episode_rewards.mean().item()
            else:
                max_return = 0.0
                avg_return = 0.0

            if makespans.numel() > 0:
                min_makespan = makespans.min().item()
                mean_makespan = makespans.mean().item()
            else:
                min_makespan = 0.0
                mean_makespan = 0.0

            logger.log_episode_metrics(
                max_return=max_return,
                avg_return=avg_return,
                max_length=tensordict_data["step_count"].max().item(),
                min_makespan=min_makespan,
                mean_makespan=mean_makespan,
                count_terminated=(
                    terminated.sum().item() if terminated is not None else 0
                ),
                step=rollout_frames,
            )
            # Save checkpoints periodically even if checkpoint_freq is not a divisor of rollout_frames
            if (
                next_checkpoint_frame is not None
                and rollout_frames >= next_checkpoint_frame
            ):
                checkpoint_logger.save_checkpoint(
                    model=policy_module,
                    metric_value=avg_return,
                    metric_name="avg_return",
                    step=rollout_frames,
                )
                next_checkpoint_frame = rollout_frames + checkpoint_freq
            pbar.update(tensordict_data.numel())

            # Evaluation using the configured evaluator
            if evaluator is not None:
                # Robust eval_freq: Ensure it is a multiple of frames_per_batch
                current_eval_freq = eval_freq
                if current_eval_freq is not None and number_frames > 0:
                    if current_eval_freq % number_frames != 0:
                        current_eval_freq = (
                            round(current_eval_freq / number_frames) * number_frames
                        )
                        if current_eval_freq == 0:
                            current_eval_freq = number_frames

                should_eval = False
                if current_eval_freq is not None and evaluator.should_evaluate(
                    rollout_frames, current_eval_freq
                ):
                    should_eval = True

                if (
                    eval_rollout_freq is not None
                    and total_rollouts % eval_rollout_freq == 0
                ):
                    should_eval = True

                if should_eval:
                    eval_metrics = evaluator.evaluate(policy_module, rollout_frames)
                    es_cfg = cfg.training.get("early_stopping", {})
                    print("Early stopping config:", es_cfg)
                    if es_cfg and eval_metrics:
                        metric_name = es_cfg.get("metric_name", "makespan")
                        patience = es_cfg.get("patience", 10)
                        mode = es_cfg.get("mode", "min")
                        target = es_cfg.get("target", None)
                        gap = es_cfg.get("gap", 0.0)

                        value = eval_metrics.get(metric_name)
                        if value is not None:
                            print(
                                f"Early stopping check for metric '{metric_name}': {value}"
                            )
                            if check_stop(
                                value, patience, mode, target, early_stop_state, gap
                            ):
                                print(
                                    f"Early stopping triggered at {rollout_frames} frames "
                                    f"for metric '{metric_name}' with value {value}."
                                )
                                rollout_frames = (
                                    cfg.training.total_frames
                                )  # To break outer loop
                                break  # Break inner loop

                    # Optional: Save best model based on evaluation metrics
                    # if "makespan" in eval_metrics:
                    #     checkpoint_logger.save_checkpoint(
                    #         model=policy_module,
                    #         metric_value=eval_metrics["makespan"],
                    #         metric_name="eval_makespan",
                    #         step=rollout_frames,
                    #     )

            # Check if we need to switch stage
            if curriculum_manager.should_update(rollout_frames):
                break  # Break inner loop to trigger update in outer loop

    pbar.close()

    # Save final model using checkpoint logger
    checkpoint_logger.save_final_checkpoint(policy_module, "policy_module_final.pt")

    # Close the logger
    logger.close()

    return policy_module
