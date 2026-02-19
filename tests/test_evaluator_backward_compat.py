"""
Test BenchmarkEvaluator generator abstraction.

This ensures that the instance generator abstraction works correctly.
"""

from jssp_core.instances.generators import RandomInstanceGenerator
from jssp_gnn.evaluator import (
    heuristic_instance_generator,
    simple_instance_generator,
)


def test_simple_generator():
    """Test simple_instance_generator functionality."""
    print("\nTesting simple_instance_generator...")

    gen = RandomInstanceGenerator(num_jobs=4, num_machines=4)
    instance_gen = simple_instance_generator(gen, reference_makespan=40.0)

    # Get some instances
    instances = []
    for i, inst in enumerate(instance_gen):
        instances.append(inst)
        if i >= 2:  # Get 3 instances
            break

    assert len(instances) == 3, "Should generate exactly 3 instances"
    assert all(inst.instance.num_jobs() == 4 for inst in instances), (
        "All should have 4 jobs"
    )
    assert all(inst.reference_makespan == 40.0 for inst in instances), (
        "All should have ref=40.0"
    )

    print(f"✓ Generated {len(instances)} instances with correct properties")


def test_evaluation_instance():
    """Test EvaluationInstance dataclass."""
    print("\nTesting EvaluationInstance...")

    from jssp_gnn.evaluator import EvaluationInstance

    gen = RandomInstanceGenerator(num_jobs=3, num_machines=3)
    instance = gen.generate()

    # Test with all fields
    eval_inst = EvaluationInstance(
        instance=instance,
        reference_makespan=30.0,
        metadata={"source": "test", "id": 123},
    )

    assert eval_inst.instance is not None
    assert eval_inst.reference_makespan == 30.0
    assert eval_inst.metadata["source"] == "test"
    print("✓ EvaluationInstance with all fields works")

    # Test with minimal fields (reference_makespan and metadata are optional)
    eval_inst_min = EvaluationInstance(instance=instance)
    assert eval_inst_min.instance is not None
    assert eval_inst_min.reference_makespan is None
    assert eval_inst_min.metadata is None
    print("✓ EvaluationInstance with minimal fields works")


def test_integration():
    """Test integration between components."""
    print("\nTesting component integration...")

    from jssp_gnn.evaluator import EvaluationInstance

    # Create a generator
    gen = RandomInstanceGenerator(
        num_jobs=5, num_machines=5, min_duration=1, max_duration=5
    )

    # Create evaluation instance generator
    instance_gen = simple_instance_generator(gen)

    # Get instances and verify structure
    count = 0
    for eval_inst in instance_gen:
        assert isinstance(eval_inst, EvaluationInstance)
        assert eval_inst.instance.num_jobs() == 5
        assert eval_inst.instance.num_machines() == 5
        count += 1
        if count >= 5:
            break

    assert count == 5, f"Expected 5 instances, got {count}"
    print(f"✓ Successfully iterated through {count} instances")


def test_heuristic_generator():
    """Test heuristic_instance_generator functionality."""
    print("\nTesting heuristic_instance_generator...")

    gen = RandomInstanceGenerator(
        num_jobs=4, num_machines=4, min_duration=1, max_duration=10
    )
    instance_gen = heuristic_instance_generator(gen, heuristic_name="mwr")

    # Get some instances
    instances = []
    for i, inst in enumerate(instance_gen):
        instances.append(inst)
        if i >= 2:  # Get 3 instances
            break

    assert len(instances) == 3, "Should generate exactly 3 instances"
    assert all(inst.instance.num_jobs() == 4 for inst in instances), (
        "All should have 4 jobs"
    )
    assert all(inst.reference_makespan is not None for inst in instances), (
        "All should have computed reference makespan"
    )
    assert all(inst.metadata["heuristic_name"] == "mwr" for inst in instances), (
        "All should have heuristic name in metadata"
    )

    # Verify that reference makespans are reasonable
    for inst in instances:
        assert inst.reference_makespan > 0, "Reference makespan should be positive"
        print(f"  Instance makespan (MWR): {inst.reference_makespan}")

    print(f"✓ Generated {len(instances)} instances with heuristic reference makespans")


if __name__ == "__main__":
    print("=" * 70)
    print("BenchmarkEvaluator Generator Abstraction Tests")
    print("=" * 70)

    test_generator_abstraction()
    test_simple_generator()
    test_evaluation_instance()
    test_integration()
    test_heuristic_generator()

    print("\n" + "=" * 70)
    print("✅ All generator abstraction tests passed!")
    print("=" * 70)
