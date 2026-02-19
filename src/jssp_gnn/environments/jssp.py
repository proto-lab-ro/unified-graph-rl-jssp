from typing import Any

import numpy as np
import torch
from tensordict import TensorDict
from torchrl.data import Categorical, Composite, DiscreteTensorSpec, UnboundedContinuous
from torchrl.envs import EnvBase

from jssp_core.domain import JobSelectorType
from jssp_core.environments.jssp import JSSPEnv
from jssp_core.instances import JSSPInstance


class GraphMatrixEnv(EnvBase):
    def __init__(
        self,
        instance: JSSPInstance,
        env_kwargs: dict,
        device: torch.device | None = None,
    ):
        if instance is None:
            raise ValueError("Instance cannot be None")
        if not isinstance(env_kwargs, dict):
            raise ValueError("env_kwargs must be a dictionary")
        self.device = device
        super().__init__(device=device, batch_size=[])
        # Override action spec to be based on number of operations

        self.env = JSSPEnv(instance=instance, **env_kwargs)
        self.num_jobs = self.env.num_jobs
        self.num_operations = self.env.num_operations
        self.num_machines = self.env.num_machines
        self.env_type = self.env.environment_type
        self.observation_type = self.env.observation_provider.observation_type

        self._make_spec()

        # self.action_spec = DiscreteTensorSpec(
        #     n=self.env.num_operations, dtype=torch.int32
        # )
        # self.env.job_selector_type = JobSelectorType.OPERATION

    def set_masking_enabled(self, enabled: bool):
        self.env.set_masking_enabled(enabled)

    def _convert_observation(self, dict_obs: Any):
        """
        Convert observation from numpy/dict to torch tensors.
        """
        if isinstance(dict_obs, np.ndarray):
            return torch.from_numpy(dict_obs)

        if isinstance(dict_obs, dict):
            return {
                k: torch.from_numpy(v) if isinstance(v, np.ndarray) else v
                for k, v in dict_obs.items()
            }

        return dict_obs

    def _make_spec(self):
        """
        Creates and sets the specifications for observation, reward, done, and action spaces.

        The observation_spec combines PyG graph data with the action space
        dynamically sized based on the number of operations.
        """
        if self.env.job_selector_type == JobSelectorType.JOB:
            mask_shape = torch.Size([self.env.num_jobs])
        elif self.env.job_selector_type == JobSelectorType.OPERATION:
            mask_shape = torch.Size([self.env.num_operations])
        else:
            raise ValueError(
                f"During _make_spec() Unknown job_selector_type  {self.env.job_selector_type}"
            )

        self.observation_spec = Composite(
            {
                "observation": self.env.observation_provider.get_observation_space_trl(),
                "mask": Categorical(
                    n=2,
                    shape=mask_shape,
                    dtype=torch.bool,
                    device=self.device,
                ),
                "makespan": UnboundedContinuous(
                    1, dtype=torch.float32, device=self.device
                ),
            }
        )
        self.reward_spec = UnboundedContinuous(1)
        self.done_spec = Categorical(n=2, shape=torch.Size([1]), dtype=torch.bool)
        self.done_spec = Composite(
            {
                "done": Categorical(
                    n=2,
                    shape=torch.Size((1,)),
                    dtype=torch.bool,
                    device=self.device,
                ),
                "terminated": Categorical(
                    n=2,
                    shape=torch.Size((1,)),
                    dtype=torch.bool,
                    device=self.device,
                ),
                "truncated": Categorical(
                    n=2,
                    shape=torch.Size((1,)),
                    dtype=torch.bool,
                    device=self.device,
                ),
            },
        )
        # Dynamic action space based on number of jobs (not operations)
        # Use Categorical spec - we'll handle one-hot conversion in ProbabilisticActor

        if self.env.job_selector_type == JobSelectorType.JOB:
            self.action_spec = DiscreteTensorSpec(n=self.num_jobs, dtype=torch.int32)
        elif self.env.job_selector_type == JobSelectorType.OPERATION:
            self.action_spec = DiscreteTensorSpec(
                n=self.num_operations, dtype=torch.int32
            )
        else:
            raise ValueError(
                f"During _make_spec() Unknown job_selector_type  {self.env.job_selector_type}"
            )
        self.cached_reset_output_zero = self.observation_spec.zero()
        self.cached_reset_output_zero.update(self.output_spec["full_done_spec"].zero())

        self.cached_step_output_zero = self.observation_spec.zero()
        self.cached_step_output_zero.update(self.output_spec["full_reward_spec"].zero())
        self.cached_step_output_zero.update(self.output_spec["full_done_spec"].zero())

    def _step(self, tensordict: TensorDict):
        """
        Execute a step in the environment.

        Args:
            tensordict: Contains the action tensors

        Returns:
            TensorDict with next observation, reward, and termination flags
        """

        action = tensordict.get("action").item()  # operation_id

        if self.env.job_selector_type == JobSelectorType.JOB:
            job = action
        elif self.env.job_selector_type == JobSelectorType.OPERATION:
            job = self.env.schedule.flat_index_to_job_op(action)[0]

        dict_obs, reward, terminated, truncated, info = self.env.step(job)
        reward = float(reward)

        graph_obs = self._convert_observation(dict_obs)

        makespan = self.env.schedule.get_makespan()
        mask = info["action_mask"]

        tensordict_out = self.cached_step_output_zero.clone()
        tensordict_out["observation"] = graph_obs
        tensordict_out["makespan"] = torch.tensor([makespan], dtype=torch.float32)
        tensordict_out["mask"] = torch.tensor(mask, dtype=torch.bool)
        tensordict_out["reward"] = reward
        tensordict_out["done"] = truncated or terminated
        tensordict_out["truncated"] = truncated
        tensordict_out["terminated"] = terminated

        return tensordict_out

    def _reset(self, tensordict=None, **kwargs):
        dict_obs, info = self.env.reset(**kwargs)

        makespan = self.env.schedule.get_makespan()
        mask = info["action_mask"]

        graph_obs = self._convert_observation(dict_obs)

        tensordict_out = self.cached_reset_output_zero.clone()
        tensordict_out["observation"] = graph_obs
        tensordict_out["makespan"] = torch.tensor([makespan], dtype=torch.float32)
        tensordict_out["mask"] = torch.tensor(mask, dtype=torch.bool)

        return tensordict_out

    def mask_rand_step(self, tensordict: TensorDict) -> TensorDict:
        mask = tensordict["mask"]

        # get random action
        valid_indices = torch.where(mask)[0]

        if len(valid_indices) == 0:
            # Fallback: if no valid actions, return first action
            action = torch.tensor(0, dtype=torch.int32)
        else:
            # Randomly select one valid action
            random_idx = torch.randint(0, len(valid_indices), (1,))
            action = valid_indices[random_idx].squeeze()

        return TensorDict(
            {"action": action},
            batch_size=tensordict.batch_size,
        )

    def _set_seed(self, seed):
        torch.manual_seed(seed)


if __name__ == "__main__":
    env = GraphMatrixEnv(
        instance=JSSPInstance.from_file("jssp_instances/ft06"),
        env_kwargs={
            "observation_provider": "lb_bipartite_gnn",
            "observation_kwargs": {
                "time_norm": 55,
                "max_edges": 102,
                "self_loop": False,
            },
        },
    )
    env.reset()
    print(env)
    import tqdm

    for _ in tqdm.tqdm(range(500), desc="Evaluating episodes"):
        env.reset()
        rollout = env.rollout(100, env.mask_rand_step, auto_reset=True)
        print(rollout)
