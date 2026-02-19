import gymnasium as gym
from gymnasium import spaces

from jssp_core.domain import (
    EnvironmentType,
    JobSelectorType,
    ObservationProvider,
    RewardFunction,
)
from jssp_core.instances import (
    InstanceGenerator,
    InstanceGeneratorLike,
    JSSPInstance,
    ensure_instance_generator,
)
from jssp_core.observation_providers import (
    get_observation_provider,
)
from jssp_core.reward_functions import get_reward_function
from jssp_core.schedule import Schedule


class JSSPEnv(gym.Env):
    """
    Gymnasium environment for Job Shop Scheduling Problem (JSSP).
    Uses the Schedule class for core scheduling logic.
    """

    def __init__(
        self,
        instance: JSSPInstance | str | dict,
        max_episode_steps: int = 500,
        random_instance: bool = True,
        reward_function: str | RewardFunction = "sparse_makespan",
        reward_kwargs: dict | None = None,
        observation_provider: str | ObservationProvider = "default",
        observation_kwargs: dict | None = None,
        job_selector_type: JobSelectorType = JobSelectorType.OPERATION,
        instance_generator: InstanceGeneratorLike | None = None,
        instance_generator_kwargs: dict | None = None,
        invalid_action_penalty: float = -1.0,
    ):
        super().__init__()
        self.original_instance = instance
        self.random_instance = random_instance
        self.max_episode_steps = max_episode_steps
        self.reward_kwargs = reward_kwargs or {}
        self.observation_kwargs = observation_kwargs or {}
        self.job_selector_type = job_selector_type
        self.environment_type = EnvironmentType.SINGLE_AGENT
        self._instance_generator: InstanceGenerator | None = None
        self._instance_generator_spec = instance_generator
        self._instance_generator_kwargs = dict(instance_generator_kwargs or {})

        self.masking_enabled = True
        self.invalid_action_penalty = invalid_action_penalty

        # Set up reward function
        self.truncate_if_invalid = self.reward_kwargs.get("truncate_if_invalid", False)

        if isinstance(reward_function, str):
            self.reward_function = get_reward_function(
                reward_function,
                initial_schedule=Schedule(instance),
                **self.reward_kwargs,
            )
        elif isinstance(reward_function, RewardFunction):
            self.reward_function = reward_function
        else:
            raise ValueError(
                "reward_function must be a string name or RewardFunction instance"
            )

        # Initialize with the provided instance to get dimensions
        self.schedule = Schedule(instance)
        self.num_jobs = self.schedule.num_jobs
        self.num_operations = self.schedule.num_operations
        self.num_machines = self.schedule.num_machines

        if self.random_instance:
            self._instance_generator = ensure_instance_generator(
                self._instance_generator_spec,
                num_jobs=self.num_jobs,
                num_machines=self.num_machines,
                generator_kwargs=self._instance_generator_kwargs,
            )

        # Set up observation provider
        if isinstance(observation_provider, str):
            self.observation_provider = get_observation_provider(
                observation_provider, self.schedule, **self.observation_kwargs
            )
        elif isinstance(observation_provider, ObservationProvider):
            self.observation_provider = observation_provider
        else:
            raise ValueError(
                "observation_provider must be a string name or ObservationProvider instance"
            )

        # Gym-specific attributes
        self.step_count = 0
        self.initial_est_makespan = None

        # Action space: index of job to schedule next operation
        if self.job_selector_type == JobSelectorType.OPERATION:
            self.action_space = spaces.Discrete(self.num_operations)

        elif self.job_selector_type == JobSelectorType.JOB:
            self.action_space = spaces.Discrete(self.num_jobs)

        else:
            raise ValueError(
                f"During action_space() Unknown job_selector_type  {self.job_selector_type}"
            )

        # Observation space from the provider
        self.observation_space = self.observation_provider.get_observation_space()

        self.reset()

    @property
    def instance(self):
        """Get the current instance (for compatibility with existing code)"""
        return self.schedule.instance

    @instance.setter
    def instance(self, new_instance):
        """Set a new instance and reinitialize the schedule"""
        self.original_instance = new_instance
        self.schedule = Schedule(new_instance)
        # Reset observation provider with new schedule
        self.observation_provider.reset(self.schedule)

    @property
    def scheduled(self):
        """Get the scheduled operations (for compatibility with existing code)"""
        return self.schedule.scheduled

    @property
    def job_next_op(self):
        """Get job next operation indices (for compatibility with existing code)"""
        return self.schedule.job_next_op

    @property
    def job_ready_time(self):
        """Get job ready times (for compatibility with existing code)"""
        return self.schedule.job_ready_time

    @property
    def machine_ready_time(self):
        """Get machine ready times (for compatibility with existing code)"""
        return self.schedule.machine_ready_time

    @property
    def eligible_operations(self):
        """Get eligible operations (for compatibility with existing code)"""
        return self.schedule.eligible_operations

    def _get_info(self) -> dict:
        """Get additional info about the environment state"""

        return {
            "step_count": self.step_count,
            "initial_est_makespan": self.initial_est_makespan,
            "action_mask": self.action_masks(),
        }

    def reset(self, seed: int | None = None, options: dict | None = None):
        """Reset the environment to initial state"""
        super().reset(seed=seed)

        # Generate new random instance if requested
        if self.random_instance:
            if self._instance_generator is None:
                self._instance_generator = ensure_instance_generator(
                    self._instance_generator_spec,
                    num_jobs=self.num_jobs,
                    num_machines=self.num_machines,
                    generator_kwargs=self._instance_generator_kwargs,
                )
            new_instance = self._instance_generator.generate()
            self._validate_instance_dimensions(new_instance)
            self.schedule = Schedule(new_instance)
            # Maintain backward compatibility for reward function name
            reward_func_name = self.get_reward_function_name()
            self.reward_function = get_reward_function(
                reward_func_name,
                initial_schedule=self.schedule,
                **self.reward_kwargs,
            )
        else:
            # Reset the schedule with original instance
            self.schedule = Schedule(self.original_instance)

        # Reset observation provider with new schedule
        self.observation_provider.reset(self.schedule)

        # Reset gym-specific state
        self.step_count = 0

        return self._get_obs(), self._get_info()

    def _get_obs(self):
        """Get the current observation using the configured observation provider"""
        return self.observation_provider.get_observation(self.schedule)

    def _get_eligible(self):
        """Get list of eligible jobs (for compatibility with existing code)"""
        return self.schedule.get_eligible_jobs()

    def estimate_completion_time(self):
        """Estimate completion time (for compatibility with existing code)"""
        return self.schedule.estimate_completion_time()

    def _calculate_reward(
        self, state_before: Schedule, state_after: Schedule, action_valid: bool = True
    ) -> float:
        """Calculate reward using the configured reward function"""
        reward_data = {
            "state_before": state_before,
            "state_after": state_after,
            "is_complete": self.schedule.is_complete(),
            "step_count": self.step_count,
            "max_episode_steps": self.max_episode_steps,
            "schedule": self.schedule,
            "action_valid": action_valid,
        }

        return self.reward_function.calculate_reward(reward_data)

    def step(self, action_job: int):
        """
        Take a step in the environment by scheduling an operation

        Args:
            action_job: ID of the job whose next operation should be scheduled

        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        self.step_count += 1

        # Check for episode timeout
        if self.step_count > self.max_episode_steps:
            return self._get_obs(), 0.0, False, True, self._get_info()

        # Validate action
        if not (0 <= action_job < self.num_jobs):
            raise ValueError(
                f"Invalid job index {action_job}, must be in [0, {self.num_jobs - 1}]"
            )

        state_before = self.schedule.copy()

        # Schedule the operation
        success = self.schedule.schedule_job(action_job)

        if not success:
            if not self.masking_enabled:
                return (
                    self._get_obs(),
                    self.invalid_action_penalty,
                    False,
                    False,
                    self._get_info(),
                )

            raise ValueError(
                f"Tried to schedule non-permitted operation {action_job}, "
            )

        state_after = self.schedule

        # Calculate reward
        reward = self._calculate_reward(state_before, state_after)

        # Check if episode is done
        terminated = self.schedule.is_complete()

        return self._get_obs(), reward, terminated, False, self._get_info()

    def get_schedule_copy(self) -> Schedule:
        """Get a copy of the current schedule for external use"""
        return self.schedule.copy()

    def set_schedule_state(self, schedule: Schedule):
        """Set the environment to a specific schedule state"""
        if (
            schedule.num_jobs != self.num_jobs
            or schedule.num_machines != self.num_machines
            or schedule.num_operations != self.num_operations
        ):
            raise ValueError("Schedule dimensions don't match environment")

        self.schedule = schedule.copy()

    def validate_current_schedule(self) -> tuple[bool, list[str]]:
        """Validate the current schedule state"""
        return self.schedule.validate_schedule()

    def set_reward_function(
        self,
        reward_function: str | RewardFunction,
        reward_kwargs: dict | None = None,
    ):
        """
        Change the reward function during runtime

        Args:
            reward_function: Either a string name or RewardFunction instance
            reward_kwargs: Optional kwargs for reward function constructor
        """
        if reward_kwargs is None:
            reward_kwargs = {}

        if isinstance(reward_function, str):
            self.reward_function = get_reward_function(
                reward_function, initial_schedule=self.schedule, **reward_kwargs
            )
        elif isinstance(reward_function, RewardFunction):
            self.reward_function = reward_function
        else:
            raise ValueError(
                "reward_function must be a string name or RewardFunction instance"
            )

    def get_reward_function_name(self) -> str:
        """Get the name of the current reward function"""
        return self.reward_function.name

    def set_observation_provider(
        self,
        observation_provider: str | ObservationProvider,
        observation_kwargs: dict | None = None,
    ):
        """
        Change the observation provider during runtime.

        Args:
            observation_provider: Either a string name or ObservationProvider instance
            observation_kwargs: Optional kwargs for observation provider constructor
        """
        if observation_kwargs is None:
            observation_kwargs = {}

        if isinstance(observation_provider, str):
            self.observation_provider = get_observation_provider(
                observation_provider, self.schedule, **observation_kwargs
            )
        elif isinstance(observation_provider, ObservationProvider):
            self.observation_provider = observation_provider
        else:
            raise ValueError(
                "observation_provider must be a string name or ObservationProvider instance"
            )

        # Update observation space
        self.observation_space = self.observation_provider.get_observation_space()

    def get_observation_provider_name(self) -> str:
        """Get the name of the current observation provider"""
        return self.observation_provider.name

    def set_masking_enabled(self, enabled: bool):
        """Enable or disable action masking."""
        self.masking_enabled = enabled

    def action_masks(self) -> list[bool] | list[int]:
        """
        Generate a mask for valid actions based on eligible jobs.
        or for valid operations.

        Returns a binary mask where 1 indicates the job can be scheduled next.
        """
        if not self.masking_enabled:
            return [True] * self.action_space.n

        if self.job_selector_type == JobSelectorType.JOB:
            mask = self.schedule.get_valid_job_mask()
        elif self.job_selector_type == JobSelectorType.OPERATION:
            mask = self.schedule.get_valid_operation_mask()
        else:
            raise ValueError(f"Unknown job_selector_type: {self.job_selector_type}")
        return mask

    def _validate_instance_dimensions(self, instance: JSSPInstance) -> None:
        """Ensure generated instances fit the configured action/observation spaces."""
        num_jobs = len(instance)
        num_operations = sum(len(job) for job in instance)
        unique_machines = {machine for job in instance for machine, _ in job}

        if num_jobs != self.num_jobs:
            raise ValueError(
                f"Generated instance has {num_jobs} jobs, expected {self.num_jobs}"
            )

        if len(unique_machines) != self.num_machines:
            raise ValueError(
                "Generated instance uses "
                f"{len(unique_machines)} machines, expected {self.num_machines}"
            )

        if num_operations != self.num_operations:
            raise ValueError(
                f"Generated instance has {num_operations} operations, "
                f"expected {self.num_operations}"
            )
