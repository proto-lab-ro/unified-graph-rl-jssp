import random

import numpy as np
import torch

from jssp_core.instances.jssp import generate_random_jssp_instance
from jssp_core.reproducibility import set_seed


def test_python_random_reproducibility():
    """Test that python's random module is reproducible."""
    seed = 123

    set_seed(seed)
    val1 = random.random()
    val2 = random.random()

    set_seed(seed)
    val1_again = random.random()
    val2_again = random.random()

    assert val1 == val1_again
    assert val2 == val2_again

    set_seed(seed + 1)
    val1_diff = random.random()
    assert val1 != val1_diff


def test_numpy_reproducibility():
    """Test that numpy's random module is reproducible."""
    seed = 456

    set_seed(seed)
    arr1 = np.random.rand(5)

    set_seed(seed)
    arr1_again = np.random.rand(5)

    np.testing.assert_array_equal(arr1, arr1_again)

    set_seed(seed + 1)
    arr1_diff = np.random.rand(5)
    assert not np.array_equal(arr1, arr1_diff)


def test_torch_reproducibility():
    """Test that torch's random module is reproducible."""
    seed = 789

    set_seed(seed)
    tensor1 = torch.rand(5, 5)

    set_seed(seed)
    tensor1_again = torch.rand(5, 5)

    assert torch.equal(tensor1, tensor1_again)

    set_seed(seed + 1)
    tensor1_diff = torch.rand(5, 5)
    assert not torch.equal(tensor1, tensor1_diff)


def test_jssp_instance_reproducibility():
    """Test that JSSP instance generation is reproducible."""
    seed = 101112
    num_jobs = 6
    num_machines = 6

    # Generate first instance
    set_seed(seed)
    instance1 = generate_random_jssp_instance(num_jobs, num_machines)

    # Generate second instance with same seed
    set_seed(seed)
    instance2 = generate_random_jssp_instance(num_jobs, num_machines)

    # Check equality
    assert instance1 == instance2

    # Generate third instance with different seed
    set_seed(seed + 1)
    instance3 = generate_random_jssp_instance(num_jobs, num_machines)

    # Check inequality
    assert instance1 != instance3
