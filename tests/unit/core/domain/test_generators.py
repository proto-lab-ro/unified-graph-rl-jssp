"""
Unit tests for instance generators.
"""

import pytest

from jssp_core.instances.generators import (
    PredefinedInstanceGenerator,
    RandomInstanceGenerator,
)
from jssp_core.instances.jssp import JSSPInstance


@pytest.mark.unit
class TestInstanceGenerators:
    """Test various instance generator classes."""

    def test_random_instance_generator(self):
        """Test RandomInstanceGenerator produces valid instances."""
        num_jobs, num_machines = 5, 3
        generator = RandomInstanceGenerator(
            num_jobs=num_jobs,
            num_machines=num_machines,
            min_duration=1,
            max_duration=10,
        )

        # Test generate method
        instance = generator.generate(seed=42)
        assert isinstance(instance, JSSPInstance)
        assert len(instance) == num_jobs
        assert all(len(job) == num_machines for job in instance)

        # Test duration bounds
        for job in instance:
            for machine, duration in job:
                assert 1 <= duration <= 10
                assert 0 <= machine < num_machines

        # Test iterability
        instance_iter = next(generator)
        assert isinstance(instance_iter, JSSPInstance)

    def test_predefined_instance_generator_sequential(self):
        """Test PredefinedInstanceGenerator in sequential mode."""
        instances = [
            [[(0, 1), (1, 2)]],  # 1 job, 2 machines
            [[(1, 3), (0, 4)]],
        ]
        generator = PredefinedInstanceGenerator(instances=instances, mode="sequential")

        # Should loop through instances
        assert generator.generate() == instances[0]
        assert generator.generate() == instances[1]
        assert generator.generate() == instances[0]  # Loop back

    def test_predefined_instance_generator_random(self):
        """Test PredefinedInstanceGenerator in random mode."""
        instances = [[[(0, i)]] for i in range(10)]
        generator = PredefinedInstanceGenerator(instances=instances, mode="random")

        # Generate several and check they are from the set
        for _ in range(5):
            instance = generator.generate()
            assert instance in instances

    def test_truncated_normal_instance_generator(self):
        """Test TruncatedNormalInstanceGenerator."""
        from jssp_core.instances.generators import TruncatedNormalInstanceGenerator

        num_jobs, num_machines = 4, 2
        generator = TruncatedNormalInstanceGenerator(
            num_jobs=num_jobs,
            num_machines=num_machines,
            min_duration=10,
            max_duration=100,
            interval=10,
            std=5.0,
        )

        # min_duration is now in constructor
        instance = generator.generate(seed=42)
        assert isinstance(instance, JSSPInstance)
        assert len(instance) == num_jobs
