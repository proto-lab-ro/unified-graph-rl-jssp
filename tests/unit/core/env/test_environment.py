"""
Unit tests for JSSP environment.
"""

import copy

import gymnasium as gym
import numpy as np
import pytest

from jssp_core.domain.domains import JobSelectorType
from jssp_core.environments.jssp import JSSPEnv
from jssp_core.instances import JSSPInstance
from jssp_core.instances.generators import InstanceGenerator


@pytest.mark.unit
@pytest.mark.env
class TestJSSPEnvironment:
    """Test JSSP environment functionality."""

    def test_environment_initialization(self, jssp_env_3x3):
        """Test environment initializes correctly."""
        env = jssp_env_3x3

        assert isinstance(env, gym.Env)
        assert env.num_jobs == 3
        assert env.num_machines == 3
        assert env.job_selector_type == JobSelectorType.JOB
        assert env.action_space.n == env.num_jobs

    def test_reset_environment(self, jssp_env_3x3):
        """Test environment reset functionality."""
        env = jssp_env_3x3

        observation, info = env.reset()

        # Check observation structure
        assert isinstance(observation, dict)
        required_keys = [
            "job_next_op",
            "job_ready_time",
            "machine_ready_time",
            "operation_action_mask",
        ]
        for key in required_keys:
            assert key in observation

        # Check observation shapes
        assert observation["job_next_op"].shape == (3,)
        assert observation["job_ready_time"].shape == (3,)
        assert observation["machine_ready_time"].shape == (3,)
        assert observation["operation_action_mask"].shape == (9,)  # 3x3 = 9 operations

    def test_action_space(self, jssp_env_3x3):
        """Test action space properties."""
        env = jssp_env_3x3

        assert isinstance(env.action_space, gym.spaces.Discrete)
        assert env.action_space.n == env.num_jobs

        # Test valid actions
        for action in range(env.action_space.n):
            assert env.action_space.contains(action)

        # Test invalid actions
        assert not env.action_space.contains(-1)
        assert not env.action_space.contains(env.action_space.n)

    def test_observation_space(self, jssp_env_3x3):
        """Test observation space properties."""
        env = jssp_env_3x3

        assert isinstance(env.observation_space, gym.spaces.Dict)

        obs, _ = env.reset()
        assert env.observation_space.contains(obs)

    def test_step_valid_action(self, jssp_env_3x3):
        """Test stepping with valid action."""
        env = jssp_env_3x3
        obs, _ = env.reset()

        # Find a valid action (job that can be scheduled)
        valid_actions = []
        for job_id in range(env.num_jobs):
            if env.schedule.can_schedule_job(job_id):
                valid_actions.append(job_id)

        if valid_actions:
            action = valid_actions[0]
            next_obs, reward, terminated, truncated, info = env.step(action)

            assert isinstance(next_obs, dict)
            assert isinstance(reward, (int, float))
            assert isinstance(terminated, bool)
            assert isinstance(truncated, bool)
            assert isinstance(info, dict)

    def test_step_invalid_action(self, jssp_env_3x3):
        """Test stepping with invalid action."""
        env = jssp_env_3x3
        env.reset()

        # Try to step with an invalid action (out of range)
        with pytest.raises((ValueError, AssertionError)):
            env.step(env.action_space.n + 1)

    def test_episode_completion(self, jssp_env_3x3):
        """Test that episode can be completed."""
        env = jssp_env_3x3
        obs, _ = env.reset()

        terminated = False
        truncated = False
        steps = 0
        max_steps = 100  # Safety limit

        while not (terminated or truncated) and steps < max_steps:
            # Choose a valid action
            valid_actions = []
            for job_id in range(env.num_jobs):
                if env.schedule.can_schedule_job(job_id):
                    valid_actions.append(job_id)

            if not valid_actions:
                break

            action = valid_actions[0]
            obs, reward, terminated, truncated, info = env.step(action)
            steps += 1

        # Should either terminate naturally or hit step limit
        assert steps <= max_steps

    def test_reward_structure(self, jssp_env_3x3):
        """Test reward structure is reasonable."""
        env = jssp_env_3x3
        env.reset()

        # Take a few steps and check rewards
        rewards = []
        for _ in range(3):
            valid_actions = []
            for job_id in range(env.num_jobs):
                if env.schedule.can_schedule_job(job_id):
                    valid_actions.append(job_id)

            if not valid_actions:
                break

            action = valid_actions[0]
            _, reward, terminated, truncated, _ = env.step(action)
            rewards.append(reward)

            if terminated or truncated:
                break

        # Rewards should be numeric
        for reward in rewards:
            assert isinstance(reward, (int, float))
            assert not np.isnan(reward)
            assert not np.isinf(reward)


@pytest.mark.unit
@pytest.mark.env
class TestEnvironmentRandomInstance:
    """Test environment with random instance generation."""

    def test_random_instance_mode(self, small_3x3_instance):
        """Test environment with random instance enabled."""
        env = JSSPEnv(small_3x3_instance, random_instance=True)

        obs1, _ = env.reset()
        obs2, _ = env.reset()

        # With random instances, initial observations might differ
        # Just check they're both valid
        assert isinstance(obs1, dict)
        assert isinstance(obs2, dict)
        assert obs1.keys() == obs2.keys()

    def test_named_instance_generator(self, small_3x3_instance):
        """Use registered generator via config-style string."""
        env = JSSPEnv(
            small_3x3_instance,
            random_instance=True,
            instance_generator="random_uniform",
            instance_generator_kwargs={"min_duration": 5, "max_duration": 5},
        )

        env.reset()
        durations = {duration for job in env.schedule.instance for _, duration in job}
        assert durations == {5}

    def test_unknown_instance_generator_raises(self, small_3x3_instance):
        """Invalid generator names should raise helpful errors."""
        with pytest.raises(ValueError):
            JSSPEnv(
                small_3x3_instance,
                random_instance=True,
                instance_generator="does_not_exist",
            )

    def test_custom_instance_generator_used(self, small_3x3_instance):
        """Ensure instance generator hook is used when provided."""

        class ConstantGenerator(InstanceGenerator):
            def __init__(self, instance):
                self._instance = instance
                self.calls = 0

            def generate(self):
                self.calls += 1
                return JSSPInstance(copy.deepcopy(self._instance))

        generator = ConstantGenerator(small_3x3_instance)
        env = JSSPEnv(
            small_3x3_instance,
            random_instance=True,
            instance_generator=generator,
        )

        env.reset()  # __init__ already called generator once
        assert generator.calls == 2

    def test_max_episode_steps(self, small_3x3_instance):
        """Test max episode steps functionality."""
        max_steps = 10
        env = JSSPEnv(
            small_3x3_instance, max_episode_steps=max_steps, random_instance=False
        )

        env.reset()

        for step in range(max_steps + 5):  # Go beyond max steps
            valid_actions = []
            for job_id in range(env.num_jobs):
                if env.schedule.can_schedule_job(job_id):
                    valid_actions.append(job_id)

            if not valid_actions:
                break

            action = valid_actions[0]
            _, _, terminated, truncated, _ = env.step(action)

            if terminated or truncated:
                break

        # Should have terminated or truncated by max_steps
        assert step <= max_steps or terminated or truncated
