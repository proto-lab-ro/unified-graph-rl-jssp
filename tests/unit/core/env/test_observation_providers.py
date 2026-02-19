"""
Unit tests for observation providers.
"""

import numpy as np
import pytest
from gymnasium import spaces

from jssp_core.environments.jssp import JSSPEnv
from jssp_core.instances import _parse_instance
from jssp_core.observation_providers.jssp import (
    OBSERVATION_PROVIDERS,
    DefaultObservationProvider,
    MinimalObservationProvider,
    NormalizedObservationProvider,
    ObservationProvider,
    OperationLowerBoundObservationProvider,
    get_observation_provider,
)


# Test instance (small for fast testing)
TEST_INSTANCE_STR = """
3 3
0 3 1 2 2 4
1 1 2 5 0 2
0 1 1 3 2 6
"""


@pytest.fixture
def test_instance():
    """Provide a test JSSP instance."""
    return _parse_instance(TEST_INSTANCE_STR.strip())


@pytest.fixture
def test_env(test_instance):
    """Provide a test environment with default observation provider."""
    return JSSPEnv(instance=test_instance, random_instance=False)


@pytest.mark.unit
class TestObservationProviders:
    """Test observation provider functionality."""

    def test_get_observation_provider_factory(self, test_instance):
        """Test the observation provider factory function."""
        from jssp_core.schedule import Schedule

        schedule = Schedule(test_instance)

        # Test getting each registered provider
        for name in OBSERVATION_PROVIDERS.keys():
            provider = get_observation_provider(name, schedule)
            assert isinstance(provider, ObservationProvider)
            assert provider.name == name

        # Test invalid provider name
        with pytest.raises(ValueError, match="Unknown component 'invalid_provider'"):
            get_observation_provider("invalid_provider", schedule)

    def test_default_observation_provider(self, test_instance):
        """Test default observation provider matches original behavior."""
        from jssp_core.schedule import Schedule

        schedule = Schedule(test_instance)
        provider = DefaultObservationProvider(schedule)

        # Test observation space
        obs_space = provider.get_observation_space()
        assert isinstance(obs_space, spaces.Dict)

        expected_keys = {
            "job_next_op",
            "job_ready_time",
            "machine_ready_time",
            "operation_action_mask",
        }
        assert set(obs_space.spaces.keys()) == expected_keys

        # Test observation generation
        obs = provider.get_observation(schedule)
        assert set(obs.keys()) == expected_keys
        assert obs["job_next_op"].shape == (3,)
        assert obs["job_ready_time"].shape == (3,)
        assert obs["machine_ready_time"].shape == (3,)
        assert obs["operation_action_mask"].shape == (9,)  # 3x3 operations

        # Test observation validation
        assert provider.validate_observation(obs)

    def test_minimal_observation_provider(self, test_instance):
        """Test minimal observation provider."""
        from jssp_core.schedule import Schedule

        schedule = Schedule(test_instance)
        provider = MinimalObservationProvider(schedule)

        # Test observation space
        obs_space = provider.get_observation_space()
        expected_keys = {"job_ready_time", "operation_action_mask"}
        assert set(obs_space.spaces.keys()) == expected_keys

        # Test observation generation
        obs = provider.get_observation(schedule)
        assert set(obs.keys()) == expected_keys
        assert obs["job_ready_time"].shape == (3,)
        assert obs["operation_action_mask"].shape == (3,)  # job-level mask

        # Test observation validation
        assert provider.validate_observation(obs)

    def test_normalized_provider_with_fixed_horizon(self, test_instance):
        """Test normalized provider with fixed time horizon."""
        from jssp_core.schedule import Schedule

        schedule = Schedule(test_instance)
        provider = NormalizedObservationProvider(schedule, max_time_horizon=100)

        obs = provider.get_observation(schedule)

        # With fixed horizon, all time values should be <= 1.0
        assert np.all(obs["job_ready_time"] <= 1.0)
        assert np.all(obs["machine_ready_time"] <= 1.0)

    def test_provider_reset(self, test_instance):
        """Test provider reset functionality."""
        from jssp_core.schedule import Schedule

        # Create provider with initial schedule
        schedule1 = Schedule(test_instance)
        provider = DefaultObservationProvider(schedule1)

        initial_num_jobs = provider.num_jobs

        # Create new instance with different dimensions
        new_instance_str = """
2 2
0 1 1 2
1 3 0 4
"""
        new_instance = _parse_instance(new_instance_str.strip())
        schedule2 = Schedule(new_instance)

        # Reset provider
        provider.reset(schedule2)

        # Check dimensions updated
        assert provider.num_jobs != initial_num_jobs
        assert provider.num_jobs == 2
        assert provider.num_machines == 2

        # Test observation generation still works
        obs = provider.get_observation(schedule2)
        assert obs["job_next_op"].shape == (2,)

    def test_operation_lower_bound_observation_provider(self, test_instance):
        """Test operation lower bound observation provider."""
        from jssp_core.schedule import Schedule

        schedule = Schedule(test_instance)

        # Test with normalization (default)
        provider = OperationLowerBoundObservationProvider(schedule)
        assert provider.name == "operation_lower_bound"

        # Test observation space
        obs_space = provider.get_observation_space()
        assert isinstance(obs_space, spaces.Dict)
        expected_keys = {
            "operation_lower_bounds",
            "operation_scheduled",
            "operation_action_mask",
        }
        assert set(obs_space.spaces.keys()) == expected_keys

        # Check shapes - should match total number of operations (3 jobs × 3 ops each = 9)
        assert obs_space["operation_lower_bounds"].shape == (9,)
        assert obs_space["operation_scheduled"].shape == (9,)
        assert obs_space["operation_action_mask"].shape == (9,)

        # Check data types
        assert obs_space["operation_lower_bounds"].dtype == np.float32
        assert obs_space["operation_scheduled"].dtype == np.int8
        assert obs_space["operation_action_mask"].dtype == np.int8

        # Test initial observation (no operations scheduled)
        obs = provider.get_observation(schedule)
        assert set(obs.keys()) == expected_keys

        # All operations should be unscheduled initially
        assert np.all(obs["operation_scheduled"] == 0)

        # Lower bounds should be normalized (between 0 and 1)
        assert np.all(obs["operation_lower_bounds"] >= 0)
        assert np.all(obs["operation_lower_bounds"] <= 1)

        # Only first operations of each job should be eligible initially
        assert np.sum(obs["operation_action_mask"]) == 3  # 3 jobs

        # Test observation validation
        assert provider.validate_observation(obs)

        # Test after scheduling some operations
        eligible_jobs = schedule.get_eligible_jobs()
        assert len(eligible_jobs) > 0

        # Schedule first eligible job
        first_job = eligible_jobs[0]
        success = schedule.schedule_job(first_job)
        assert success

        # Get new observation
        obs_after = provider.get_observation(schedule)

        # One operation should now be scheduled
        assert np.sum(obs_after["operation_scheduled"]) == 1

        # The scheduled operation should have action_mask = 0
        scheduled_ops = np.where(obs_after["operation_scheduled"] == 1)[0]
        assert len(scheduled_ops) == 1
        scheduled_op_idx = scheduled_ops[0]
        assert obs_after["operation_action_mask"][scheduled_op_idx] == 0

    def test_operation_lower_bound_provider_without_normalization(self, test_instance):
        """Test operation lower bound provider without normalization."""
        from jssp_core.schedule import Schedule

        schedule = Schedule(test_instance)
        provider = OperationLowerBoundObservationProvider(schedule, normalize=False)

        # Observation space should allow larger values
        obs_space = provider.get_observation_space()
        assert obs_space["operation_lower_bounds"].high[0] == np.inf

        # Test observation
        obs = provider.get_observation(schedule)

        # Lower bounds should not be normalized (can be > 1)
        # Since we're not normalizing, values depend on actual problem instance
        assert np.all(obs["operation_lower_bounds"] >= 0)
        # Some values might be > 1 since they're not normalized

    def test_operation_lower_bound_provider_schedule_consistency(self, test_instance):
        """Test that the provider correctly tracks scheduling state."""
        from jssp_core.schedule import Schedule

        schedule = Schedule(test_instance)
        provider = OperationLowerBoundObservationProvider(schedule)

        # Schedule multiple operations and verify consistency
        scheduled_count = 0
        while not schedule.is_complete() and scheduled_count < 5:
            obs_before = provider.get_observation(schedule)

            # Verify scheduled count matches
            actual_scheduled = np.sum(obs_before["operation_scheduled"])
            assert actual_scheduled == scheduled_count

            # Schedule next eligible operation
            eligible_jobs = schedule.get_eligible_jobs()
            if not eligible_jobs:
                break

            success = schedule.schedule_job(eligible_jobs[0])
            if success:
                scheduled_count += 1

        # Final check
        obs_final = provider.get_observation(schedule)
        assert np.sum(obs_final["operation_scheduled"]) == scheduled_count

    def test_operation_lower_bound_provider_bounds_consistency(self, test_instance):
        """Test that lower bounds are consistent with schedule state."""
        from jssp_core.schedule import Schedule

        schedule = Schedule(test_instance)
        provider = OperationLowerBoundObservationProvider(schedule, normalize=False)

        # Get initial lower bounds
        obs_initial = provider.get_observation(schedule)
        initial_bounds = obs_initial["operation_lower_bounds"].copy()

        # Schedule one operation
        eligible_jobs = schedule.get_eligible_jobs()
        if eligible_jobs:
            schedule.schedule_job(eligible_jobs[0])

            # Get updated bounds
            obs_after = provider.get_observation(schedule)
            after_bounds = obs_after["operation_lower_bounds"]

            # Bounds should be valid (non-negative)
            assert np.all(after_bounds >= 0)

            # The structure should be consistent
            assert len(after_bounds) == len(initial_bounds)


@pytest.mark.unit
class TestEnvironmentWithObservationProviders:
    """Test environment integration with observation providers."""

    def test_environment_with_different_providers(self, test_instance):
        """Test creating environments with different observation providers."""
        for provider_name in OBSERVATION_PROVIDERS.keys():
            # Skip AGV-specific providers as they require TransportSchedule (which is removed)
            if provider_name.startswith("agv"):
                continue

            env = JSSPEnv(
                instance=test_instance,
                random_instance=False,
                observation_provider=provider_name,
            )

            # Test reset
            obs, info = env.reset()
            assert env.observation_space.contains(obs)
            assert env.get_observation_provider_name() == provider_name

            # Test step
            eligible_jobs = env._get_eligible()
            if eligible_jobs:
                action = eligible_jobs[0]
                obs, reward, done, truncated, info = env.step(action)
                assert env.observation_space.contains(obs)

    def test_switching_observation_providers(self, test_env):
        """Test switching observation providers at runtime."""
        # Initial provider
        assert test_env.get_observation_provider_name() == "default"

        initial_obs, _ = test_env.reset()
        initial_keys = set(initial_obs.keys())

        # Switch to minimal provider
        test_env.set_observation_provider("minimal")
        assert test_env.get_observation_provider_name() == "minimal"

        minimal_obs = test_env._get_obs()
        minimal_keys = set(minimal_obs.keys())

        # Observation structure should be different
        assert minimal_keys != initial_keys
        assert test_env.observation_space.contains(minimal_obs)

    def test_environment_backward_compatibility(self, test_instance):
        """Test that environments without explicit observation provider work as before."""
        # Create environment without specifying observation provider (should use default)
        env = JSSPEnv(instance=test_instance, random_instance=False)

        obs, info = env.reset()

        # Should have the same observation structure as before
        expected_keys = {
            "job_next_op",
            "job_ready_time",
            "machine_ready_time",
            "operation_action_mask",
        }
        assert set(obs.keys()) == expected_keys

        # Should be able to take steps normally
        eligible_jobs = env._get_eligible()
        if eligible_jobs:
            action = eligible_jobs[0]
            obs, reward, done, truncated, info = env.step(action)
            assert set(obs.keys()) == expected_keys


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
