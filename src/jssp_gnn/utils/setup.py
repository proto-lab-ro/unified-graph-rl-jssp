import torch
from omegaconf import DictConfig
from torchrl.collectors import SyncDataCollector
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE

from .device import get_device


def create_collector_and_buffer(
    env, policy_module, cfg: DictConfig, frames_collected=0
):
    """Create collector and replay buffer."""
    collector = SyncDataCollector(
        env,
        policy_module,
        frames_per_batch=cfg.training.frames_per_batch,
        total_frames=cfg.training.total_frames - frames_collected,
        split_trajs=cfg.training.split_trajs,
        device=get_device(),
        reset_at_each_iter=True,
    )

    replay_buffer = ReplayBuffer(
        storage=LazyTensorStorage(max_size=cfg.training.frames_per_batch),
        sampler=SamplerWithoutReplacement(shuffle=True),
        batch_size=cfg.training.sub_batch_size,
    )
    return collector, replay_buffer


def setup_ppo_training_components(policy_module, value_module, env, cfg: DictConfig):
    """Setup training components: collector, loss, optimizer, etc."""
    collector, replay_buffer = create_collector_and_buffer(
        env, policy_module, cfg, frames_collected=0
    )

    advantage_module = GAE(
        gamma=cfg.training.gamma,
        lmbda=cfg.training.lmbda,
        value_network=value_module,
        average_gae=cfg.training.average_gae,
        deactivate_vmap=True,
        device=get_device(),
    )

    loss_module = ClipPPOLoss(
        actor_network=policy_module,
        critic_network=value_module,
        clip_epsilon=cfg.training.clip_epsilon,
        entropy_bonus=bool(cfg.training.entropy_eps),
        entropy_coeff=cfg.training.entropy_eps,
        critic_coeff=cfg.training.critic_coef,
        loss_critic_type=cfg.training.loss_critic_type,
        normalize_advantage=cfg.training.normalize_advantage,
    )

    optimizer = torch.optim.Adam(loss_module.parameters(), cfg.training.lr)
    scheduler = None
    if cfg.training.get("use_lr_scheduler", False):
        scheduler_type = cfg.training.get("lr_scheduler_type", "linear")

        import math

        # Estimated total optimizer steps: batches_per_rollout * num_epochs * num_rollouts
        total_rollouts = math.ceil(
            cfg.training.total_frames / cfg.training.frames_per_batch
        )
        # once per epoch
        decay_steps = cfg.training.get(
            "lr_decay_steps",
            cfg.training.num_epochs * total_rollouts,
        )
        print("lr_decay_steps:", decay_steps)

        if scheduler_type == "linear":
            # Linear decay to 0
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer,
                lr_lambda=lambda step: max(0.0, 1.0 - step / max(1, decay_steps)),
            )
        elif scheduler_type == "cosine":
            # Cosine annealing - smooth decay
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=decay_steps,
                eta_min=cfg.training.get("lr_min", 0.0),
            )

        elif scheduler_type == "exponential":
            # Exponential decay
            gamma = cfg.training.get("lr_gamma", 0.99)
            scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer,
                gamma=gamma,
            )

        elif scheduler_type == "step":
            # Step decay at intervals
            step_size = cfg.training.get("lr_step_size", decay_steps // 4)
            gamma = cfg.training.get("lr_gamma", 0.5)
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=step_size,
                gamma=gamma,
            )

        elif scheduler_type == "multistep":
            # Multiple step decays
            milestones = cfg.training.get(
                "lr_milestones",
                [decay_steps // 3, 2 * decay_steps // 3],
            )
            gamma = cfg.training.get("lr_gamma", 0.1)
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer,
                milestones=milestones,
                gamma=gamma,
            )
        elif scheduler_type == "plateau":
            # Reduce on plateau (based on metric)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="max",  # maximize reward
                factor=cfg.training.get("lr_factor", 0.5),
                patience=cfg.training.get("lr_patience", 10),
            )
        else:
            raise ValueError(f"Unknown scheduler type: {scheduler_type}")

    return collector, advantage_module, replay_buffer, loss_module, optimizer, scheduler
