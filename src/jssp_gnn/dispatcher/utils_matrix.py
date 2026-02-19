import re

from omegaconf import DictConfig
from tensordict.nn import TensorDictModule
from torchrl.envs import (
    Compose,
    ParallelEnv,
    RewardSum,
    StepCounter,
    TransformedEnv,
)
from torchrl.modules import MaskedCategorical, ProbabilisticActor, ValueOperator

from jssp_core.instances import get_instance
from jssp_gnn.environments.jssp import GraphMatrixEnv
from jssp_gnn.modules.graph_matrix import (
    HeteroSB3LikeActorMatrix,
    HeteroSB3LikeCriticMatrix,
    HeteroSharedGraphFeatureExtractorMatrix,
    SB3LikeActorMatrix,
    SB3LikeCriticMatrix,
    SharedGraphFeatureExtractorMatrix,
)
from jssp_gnn.utils import get_device


def create_models(env, cfg: DictConfig):
    """Create policy and value models with shared GNN feature extractor."""

    # Check if we are using heterogeneous GNN
    is_hetero = cfg.env.observation_provider == "lb_hetero_gnn"
    # Use observation_spec instead of accessing env.env directly (compatible with ParallelEnv)
    obs_spec = env.observation_spec["observation"]
    job_selector_type = cfg.env.job_selector_type

    if is_hetero:
        # Define metadata and input dims for Hetero GNN
        # These should match LbGnnHeteroObservationProvider
        node_types = ["operation", "machine"]
        edge_types = [
            ("operation", "precedence", "operation"),
            ("operation", "assignment", "machine"),
            ("machine", "assignment", "operation"),
        ]
        metadata = (node_types, edge_types)

        # Get input dimensions from observation spec
        input_dims = {
            "operation": obs_spec["node_feats_op"].shape[-1],
            "machine": obs_spec["node_feats_machine"].shape[-1],
        }

        shared_extractor = HeteroSharedGraphFeatureExtractorMatrix(
            metadata=metadata,
            input_dims=input_dims,
            hidden_dim=cfg.gnn_feature_extractor.get("hidden_dim"),
            k_layers=cfg.gnn_feature_extractor.get("k_layers"),
        )

        # Hetero Actor
        policy_in_keys = ["observation", "mask"]

        policy_module = TensorDictModule(
            HeteroSB3LikeActorMatrix(
                shared_extractor=shared_extractor,
                forward_type=job_selector_type,
            ),
            in_keys=policy_in_keys,
            out_keys=["logits"],
        )

        value_module = ValueOperator(
            module=HeteroSB3LikeCriticMatrix(
                shared_extractor=shared_extractor,
            ),
            in_keys=["observation"],
            out_keys=["state_value"],
        )

    else:
        # Standard Homogeneous GNN
        shared_extractor = SharedGraphFeatureExtractorMatrix(
            input_dim=obs_spec["node_feats"].shape[-1],
            hidden_dim=cfg.gnn_feature_extractor.get("hidden_dim"),
            k_layers=cfg.gnn_feature_extractor.get("k_layers"),
        )

        graph_observation_keys = [
            ("observation", "node_feats"),
            ("observation", "edge_index"),
        ]
        policy_in_keys = graph_observation_keys + ["mask"]

        policy_module = TensorDictModule(
            SB3LikeActorMatrix(
                shared_extractor=shared_extractor,
                forward_type=job_selector_type,
            ),
            in_keys=policy_in_keys,
            out_keys=["logits"],
        )

        value_module = ValueOperator(
            module=SB3LikeCriticMatrix(
                shared_extractor=shared_extractor,
            ),
            in_keys=graph_observation_keys,
            out_keys=["state_value"],
        )

    policy_module = ProbabilisticActor(
        module=policy_module,
        spec=env.action_spec,
        in_keys=[("logits"), ("mask")],
        distribution_class=MaskedCategorical,
        return_log_prob=True,
    )

    return policy_module, value_module


def load_instance_from_cfg(cfg: DictConfig):
    instance_spec = cfg.env.instance

    # Check if instance is in "NxM" format
    match = re.match(r"^(\d+)x(\d+)$", str(instance_spec))
    if match:
        num_jobs = int(match.group(1))
        num_machines = int(match.group(2))

        gen_kwargs = cfg.env.get("instance_generator_kwargs", {})

        return get_instance(
            {
                "type": "random",
                "num_jobs": num_jobs,
                "num_machines": num_machines,
                "min_duration": gen_kwargs.get("min_duration", 1),
                "max_duration": gen_kwargs.get("max_duration", 10),
                "seed": 42,
            }
        )

    return get_instance(
        {
            "type": "path",
            "path": cfg.env.instance,
        }
    )


def _make_single_env(cfg: DictConfig):
    """Helper to create a single environment instance."""
    instance = load_instance_from_cfg(cfg)

    env_kwargs = {
        "max_episode_steps": cfg.env.max_episode_steps,
        "random_instance": cfg.env.random_instance,
        "reward_function": cfg.env.reward_function,
        "reward_kwargs": cfg.env.get("reward_kwargs", {}),
        "observation_provider": cfg.env.observation_provider,
        "observation_kwargs": cfg.env.get("observation_kwargs", {}),
        "job_selector_type": cfg.env.job_selector_type,
        "instance_generator": cfg.env.get("instance_generator", None),
        "instance_generator_kwargs": cfg.env.get("instance_generator_kwargs", {}),
        "invalid_action_penalty": cfg.env.get("invalid_action_penalty", -1.0),
    }
    # Note: Device handling in ParallelEnv needs care.
    # Here we use get_device() which might be CUDA.
    # If using ParallelEnv with spawn, this should be fine as long as CUDA is initialized properly.
    # Typically, envs run on CPU in parallel and data is moved to GPU.
    # If get_device() is GPU, GraphMatrixEnv will attempt to use it.
    env = GraphMatrixEnv(
        instance=instance,
        env_kwargs=env_kwargs,
        device=get_device(),
    )
    env = TransformedEnv(
        env,
        Compose(
            StepCounter(max_steps=cfg.training.max_steps),
            RewardSum(),
        ),
    )
    return env


def create_environment(cfg: DictConfig):
    """Create and configure the environment."""
    num_envs = cfg.env.get("num_envs", 1)

    if num_envs > 1:
        # Create parallel environment
        return ParallelEnv(
            num_workers=num_envs,
            create_env_fn=lambda: _make_single_env(cfg),
        )

    return _make_single_env(cfg)
