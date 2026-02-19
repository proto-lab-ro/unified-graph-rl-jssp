"""
Unit tests for JSSP Schedule class.
"""

import numpy as np
import pytest

from jssp_core.models.operation import OperationInfo
from jssp_core.schedule import Schedule


@pytest.mark.unit
class TestSchedule:
    """Test Schedule class functionality."""

    def test_schedule_initialization(self, schedule_3x3):
        """Test schedule initializes correctly."""
        schedule = schedule_3x3

        assert schedule.num_jobs == 3
        assert schedule.num_machines == 3
        assert schedule.num_operations == 9  # 3x3

        # Check initial state
        assert schedule.job_next_op == [0, 0, 0]
        assert schedule.job_ready_time == [0, 0, 0]
        assert len(schedule.scheduled) == 0

    def test_reset_functionality(self, schedule_3x3):
        """Test schedule reset functionality."""
        schedule = schedule_3x3

        # Make some changes to the schedule
        if schedule.can_schedule_job(0):
            schedule.schedule_job(0)

        # Reset and verify
        schedule.reset()
        assert schedule.job_next_op == [0, 0, 0]
        assert schedule.job_ready_time == [0, 0, 0]
        assert len(schedule.scheduled) == 0

    def test_get_eligible_jobs(self, schedule_3x3):
        """Test getting eligible jobs."""
        schedule = schedule_3x3

        eligible = schedule.get_eligible_jobs()
        assert isinstance(eligible, list)
        assert len(eligible) > 0  # Should have eligible jobs initially

        # All jobs should be eligible initially
        assert set(eligible) == {0, 1, 2}

    def test_can_schedule_job(self, schedule_3x3):
        """Test can_schedule_job method."""
        schedule = schedule_3x3

        # Initially all jobs should be schedulable
        for job_id in range(schedule.num_jobs):
            assert schedule.can_schedule_job(job_id)

    def test_get_operation_info(self, schedule_3x3):
        """Test getting operation information."""
        schedule = schedule_3x3

        op_info = schedule.get_operation_info(0, 0)
        assert isinstance(op_info, OperationInfo)
        assert op_info.job_id == 0
        assert op_info.op_id == 0
        assert isinstance(op_info.machine, int)
        assert isinstance(op_info.duration, int)
        assert op_info.machine >= 0
        assert op_info.duration > 0

    def test_get_next_operation_info(self, schedule_3x3):
        """Test getting next operation information."""
        schedule = schedule_3x3

        next_op = schedule.get_next_operation_info(0)
        assert next_op is not None
        assert next_op.job_id == 0
        assert next_op.op_id == 0

    def test_schedule_operation(self, schedule_3x3):
        """Test scheduling an operation."""
        schedule = schedule_3x3

        initial_scheduled_count = len(schedule.scheduled)
        success = schedule.schedule_job(0)

        assert success is True
        assert len(schedule.scheduled) == initial_scheduled_count + 1
        assert schedule.job_next_op[0] == 1
        assert (0, 0) in schedule.scheduled

    def test_schedule_invalid_job(self, schedule_3x3):
        """Test scheduling invalid job."""
        schedule = schedule_3x3

        # Try to schedule non-existent job
        success = schedule.schedule_job(999)
        assert success is False

    def test_is_complete(self, schedule_3x3):
        """Test completion check."""
        schedule = schedule_3x3

        # Initially should not be complete
        assert not schedule.is_complete()

        # Schedule all operations
        steps = 0
        max_steps = 20  # Safety limit

        while not schedule.is_complete() and steps < max_steps:
            eligible = schedule.get_eligible_jobs()
            if not eligible:
                break

            success = schedule.schedule_job(eligible[0])
            if not success:
                break
            steps += 1

        # Should be complete or hit safety limit
        if steps < max_steps:
            assert schedule.is_complete()

    def test_get_makespan(self, schedule_3x3):
        """Test makespan calculation."""
        schedule = schedule_3x3

        initial_makespan = schedule.get_makespan()
        assert initial_makespan == 0

        # Schedule one operation
        if schedule.can_schedule_job(0):
            schedule.schedule_job(0)
            makespan = schedule.get_makespan()
            assert makespan > 0

    def test_get_scheduled_operations_count(self, schedule_3x3):
        """Test getting scheduled operations count."""
        schedule = schedule_3x3

        assert schedule.get_scheduled_operations_count() == 0

        if schedule.can_schedule_job(0):
            schedule.schedule_job(0)
            assert schedule.get_scheduled_operations_count() == 1

    def test_get_remaining_operations_count(self, schedule_3x3):
        """Test getting remaining operations count."""
        schedule = schedule_3x3

        total_ops = schedule.num_operations
        assert schedule.get_remaining_operations_count() == total_ops

        if schedule.can_schedule_job(0):
            schedule.schedule_job(0)
            assert schedule.get_remaining_operations_count() == total_ops - 1

    def test_estimate_completion_time(self, schedule_3x3):
        """Test completion time estimation."""
        schedule = schedule_3x3

        estimated_time = schedule.estimate_completion_time()
        assert isinstance(estimated_time, (int, float))
        assert estimated_time >= 0

    def test_get_job_progress(self, schedule_3x3):
        """Test getting job progress."""
        schedule = schedule_3x3

        completed, total = schedule.get_job_progress(0)
        assert completed == 0
        assert total == 3  # 3 operations in 3x3 instance

        if schedule.can_schedule_job(0):
            schedule.schedule_job(0)
            completed, total = schedule.get_job_progress(0)
            assert completed == 1
            assert total == 3

    def test_get_machine_utilization(self, schedule_3x3):
        """Test getting machine utilization."""
        schedule = schedule_3x3

        utilization = schedule.get_machine_utilization()
        assert isinstance(utilization, dict)

        # Initially all machines should have zero utilization
        for machine_id in range(schedule.num_machines):
            assert utilization.get(machine_id, 0) == 0


@pytest.mark.unit
class TestScheduleWithFT06:
    """Test Schedule with FT06 instance."""

    def test_ft06_properties(self, schedule_ft06):
        """Test FT06 specific properties."""
        schedule = schedule_ft06

        assert schedule.num_jobs == 6
        assert schedule.num_machines == 6
        assert schedule.num_operations == 36  # 6x6

    def test_ft06_scheduling_sequence(self, schedule_ft06):
        """Test a short scheduling sequence with FT06."""
        schedule = schedule_ft06

        # Schedule first few operations
        for _ in range(3):
            eligible = schedule.get_eligible_jobs()
            if eligible:
                success = schedule.schedule_job(eligible[0])
                assert success
            else:
                break

        assert schedule.get_scheduled_operations_count() <= 3
        assert not schedule.is_complete()  # Should not be complete after 3 ops


@pytest.mark.unit
class TestLowerBounds:
    """Test lower bound calculation functionality."""

    def test_simple_lower_bounds_unscheduled(self):
        """Test lower bounds for a simple unscheduled instance."""
        # Simple 2x2 instance for clear testing
        # Job 0: M0(3) -> M1(2)
        # Job 1: M1(4) -> M0(1)
        instance = [[(0, 3), (1, 2)], [(1, 4), (0, 1)]]
        schedule = Schedule(instance)

        bounds = schedule.get_operation_lower_bounds()

        # Job 0 operations
        assert bounds[(0, 0)] == 3.0  # First op: duration 3
        assert bounds[(0, 1)] == 5.0  # Second op: 3 + 2 = 5

        # Job 1 operations
        assert bounds[(1, 0)] == 4.0  # First op: duration 4
        assert bounds[(1, 1)] == 5.0  # Second op: 4 + 1 = 5

    def test_single_operation_lower_bound(self):
        """Test individual operation lower bound calculation."""
        instance = [
            [(0, 3), (1, 2), (2, 1)],  # Job 0: M0(3) -> M1(2) -> M2(1)
        ]
        schedule = Schedule(instance)

        # Test each operation individually
        assert schedule.get_operation_lower_bound(0, 0) == 3.0  # 3
        assert schedule.get_operation_lower_bound(0, 1) == 5.0  # 3 + 2
        assert schedule.get_operation_lower_bound(0, 2) == 6.0  # 3 + 2 + 1

    def test_lower_bounds_after_scheduling(self):
        """Test how lower bounds change after scheduling operations."""
        instance = [
            [(0, 3), (1, 2)],  # Job 0: M0(3) -> M1(2)
            [(1, 4), (0, 1)],  # Job 1: M1(4) -> M0(1)
        ]
        schedule = Schedule(instance)

        # Initial lower bounds
        initial_bounds = schedule.get_operation_lower_bounds()
        assert initial_bounds[(0, 0)] == 3.0
        assert initial_bounds[(0, 1)] == 5.0

        # Schedule first operation of Job 0 (it will start at time 0, end at time 3)
        schedule.schedule_job(0)

        # Check updated lower bounds
        updated_bounds = schedule.get_operation_lower_bounds()

        # J0O0 should now reflect actual completion time
        assert updated_bounds[(0, 0)] == 3.0  # Actual completion time

        # J0O1 should be based on actual completion of J0O0
        assert updated_bounds[(0, 1)] == 5.0  # 3 (actual completion) + 2 (duration)

    def test_lower_bounds_with_machine_conflicts(self):
        """Test lower bounds when machines have conflicts."""
        instance = [
            [(0, 2), (1, 3)],  # Job 0: M0(2) -> M1(3)
            [(0, 4), (1, 1)],  # Job 1: M0(4) -> M1(1) (both jobs use same machines)
        ]
        schedule = Schedule(instance)

        # Schedule J0O0 first (M0, duration 2)
        schedule.schedule_job(0)  # Occupies M0 from 0-2

        # Now schedule J1O0 (M0, duration 4) - should start after J0O0 finishes
        schedule.schedule_job(1)  # Should start at time 2, end at time 6

        bounds = schedule.get_operation_lower_bounds()

        # J0O0 completed at time 2
        assert bounds[(0, 0)] == 2.0

        # J0O1 should start after J0O0 completes (at time 2) and run for 3 units
        assert bounds[(0, 1)] == 5.0  # 2 + 3

        # J1O0 completed at time 6 (started at 2, duration 4)
        assert bounds[(1, 0)] == 6.0

        # J1O1 should start after J1O0 completes and run for 1 unit
        assert bounds[(1, 1)] == 7.0  # 6 + 1

    def test_lower_bounds_user_example_case(self):
        """Test the specific examples from user requirements."""
        # User example: "if the first operation need 3 time units and the second needs 2 the lb for the second is 5"
        instance = [[(0, 3), (1, 2)]]  # Job 0: M0(3) -> M1(2)
        schedule = Schedule(instance)

        bounds = schedule.get_operation_lower_bounds()
        assert bounds[(0, 0)] == 3.0  # First operation: 3 time units
        assert bounds[(0, 1)] == 5.0  # Second operation: 3 + 2 = 5

        # User example: "if the first operation was finished at 10 the lb for the second is 12"
        # Manually set the schedule state to simulate first operation finishing at 10
        schedule.scheduled[(0, 0)] = 7  # Start at 7, duration 3, so ends at 10
        schedule.job_next_op[0] = 1
        schedule.job_ready_time[0] = 10
        schedule.machine_ready_time[0] = 10

        updated_bounds = schedule.get_operation_lower_bounds()
        assert updated_bounds[(0, 0)] == 10.0  # First operation completed at 10
        assert updated_bounds[(0, 1)] == 12.0  # Second operation: 10 + 2 = 12

    def test_lower_bound_edge_cases(self):
        """Test edge cases for lower bound calculations."""
        # Single operation job
        instance = [[(0, 5)]]  # Job 0: just M0(5)
        schedule = Schedule(instance)

        bounds = schedule.get_operation_lower_bounds()
        assert bounds[(0, 0)] == 5.0

        # Single job with multiple operations
        multi_op_instance = [[(0, 2), (1, 3), (2, 1)]]
        multi_schedule = Schedule(multi_op_instance)
        multi_bounds = multi_schedule.get_operation_lower_bounds()
        assert multi_bounds[(0, 0)] == 2.0
        assert multi_bounds[(0, 1)] == 5.0  # 2 + 3
        assert multi_bounds[(0, 2)] == 6.0  # 2 + 3 + 1

    def test_lower_bound_invalid_inputs(self):
        """Test lower bound methods with invalid inputs."""
        instance = [[(0, 3), (1, 2)]]
        schedule = Schedule(instance)

        # Invalid job_id
        with pytest.raises(ValueError, match="Invalid job_id"):
            schedule.get_operation_lower_bound(-1, 0)

        with pytest.raises(ValueError, match="Invalid job_id"):
            schedule.get_operation_lower_bound(999, 0)

        # Invalid op_id
        with pytest.raises(ValueError, match="Invalid op_id"):
            schedule.get_operation_lower_bound(0, -1)

        with pytest.raises(ValueError, match="Invalid op_id"):
            schedule.get_operation_lower_bound(0, 999)

    def test_lower_bounds_consistency(self):
        """Test that lower bounds are consistent between methods."""
        instance = [
            [(0, 3), (1, 2), (2, 1)],  # Job 0
            [(1, 4), (2, 2), (0, 3)],  # Job 1
        ]
        schedule = Schedule(instance)

        # Get bounds using both methods
        all_bounds = schedule.get_operation_lower_bounds()

        # Check each operation individually
        for job_id in range(schedule.num_jobs):
            for op_id in range(len(schedule.instance[job_id])):
                individual_bound = schedule.get_operation_lower_bound(job_id, op_id)
                all_bounds_value = all_bounds[(job_id, op_id)]

                assert individual_bound == all_bounds_value, (
                    f"Inconsistent bounds for J{job_id}O{op_id}: {individual_bound} vs {all_bounds_value}"
                )

    def test_lower_bounds_progressive_scheduling(self):
        """Test lower bounds during progressive scheduling."""
        instance = [
            [(0, 2), (1, 3)],  # Job 0: M0(2) -> M1(3)
            [(1, 1), (0, 4)],  # Job 1: M1(1) -> M0(4)
        ]
        schedule = Schedule(instance)

        # Test step by step scheduling
        initial_bounds = schedule.get_operation_lower_bounds()

        # Initial bounds should be cumulative durations
        assert initial_bounds[(0, 0)] == 2.0  # M0(2)
        assert initial_bounds[(0, 1)] == 5.0  # M0(2) + M1(3)
        assert initial_bounds[(1, 0)] == 1.0  # M1(1)
        assert initial_bounds[(1, 1)] == 5.0  # M1(1) + M0(4)

        # Schedule J0O0 (M0, duration 2) - starts at 0, ends at 2
        schedule.schedule_job(0)
        bounds_after_j0o0 = schedule.get_operation_lower_bounds()

        # J0O0 should now reflect actual completion
        assert bounds_after_j0o0[(0, 0)] == 2.0  # Completed at time 2

        # J0O1 should still be based on J0O0's completion
        assert bounds_after_j0o0[(0, 1)] == 5.0  # 2 (J0O0 completion) + 3 (duration)

        # Schedule J1O0 (M1, duration 1) - starts at 0, ends at 1
        schedule.schedule_job(1)
        bounds_after_j1o0 = schedule.get_operation_lower_bounds()

        # J1O0 should reflect actual completion
        assert bounds_after_j1o0[(1, 0)] == 1.0  # Completed at time 1

        # J1O1 should be based on J1O0's completion
        assert bounds_after_j1o0[(1, 1)] == 5.0  # 1 (J1O0 completion) + 4 (duration)

        # Continue with remaining operations
        if schedule.can_schedule_job(0):  # J0O1
            schedule.schedule_job(0)

        if schedule.can_schedule_job(1):  # J1O1
            schedule.schedule_job(1)

        # Verify final bounds reflect actual schedule
        final_bounds = schedule.get_operation_lower_bounds()

        # All scheduled operations should have bounds equal to their completion times
        for (job_id, op_id), start_time in schedule.scheduled.items():
            duration = schedule.instance[job_id][op_id][1]
            expected_completion = start_time + duration
            actual_bound = final_bounds[(job_id, op_id)]

            assert actual_bound == expected_completion, (
                f"J{job_id}O{op_id}: bound {actual_bound} != completion {expected_completion}"
            )

    def test_lower_bounds_with_complex_instance(self, schedule_3x3):
        """Test lower bounds with the 3x3 fixture."""
        schedule = schedule_3x3

        # Get initial bounds
        initial_bounds = schedule.get_operation_lower_bounds()

        # Should have bounds for all operations
        assert len(initial_bounds) == schedule.num_operations

        # All bounds should be positive
        for bound in initial_bounds.values():
            assert bound > 0

        # Within each job, later operations should have higher or equal bounds
        for job_id in range(schedule.num_jobs):
            job_bounds = [
                initial_bounds[(job_id, op_id)]
                for op_id in range(len(schedule.instance[job_id]))
            ]

            # Bounds should be non-decreasing within a job
            for i in range(1, len(job_bounds)):
                assert job_bounds[i] >= job_bounds[i - 1], (
                    f"Job {job_id}: bound for op {i} ({job_bounds[i]}) < bound for op {i - 1} ({job_bounds[i - 1]})"
                )

    def test_build_precedence_edge_index_basic(self, schedule_3x3):
        """Test basic functionality of build_precedence_edge_index."""
        schedule = schedule_3x3
        edge_index = schedule.build_precedence_edge_index()

        # Check return type and shape
        assert isinstance(edge_index, np.ndarray)
        assert edge_index.dtype == np.int64
        assert edge_index.shape[0] == 2  # Should have 2 rows

        # For 3x3 instance, each job has 3 operations, so 2 edges per job = 6 total edges
        expected_edges = sum(len(job) - 1 for job in schedule.instance)
        assert edge_index.shape[1] == expected_edges

        # Check edge values are within valid node range
        num_operations = schedule.num_operations
        assert np.all(edge_index >= 0)
        assert np.all(edge_index < num_operations)

    def test_build_precedence_edge_index_structure(self, schedule_3x3):
        """Test the structure of edges in build_precedence_edge_index."""
        schedule = schedule_3x3
        edge_index = schedule.build_precedence_edge_index()

        # Create the node mapping to verify edges
        op_node_id = {}
        node_id = 0
        for job_idx, job in enumerate(schedule.instance):
            for op_idx in range(len(job)):
                op_node_id[(job_idx, op_idx)] = node_id
                node_id += 1

        # Verify that edges represent correct precedence constraints
        sources, targets = edge_index[0], edge_index[1]

        edge_set = set(zip(sources, targets, strict=False))
        expected_edges = set()

        for job_idx, job in enumerate(schedule.instance):
            for op_idx in range(len(job) - 1):
                source_node = op_node_id[(job_idx, op_idx)]
                target_node = op_node_id[(job_idx, op_idx + 1)]
                expected_edges.add((source_node, target_node))

        assert edge_set == expected_edges

    def test_build_precedence_edge_index_single_operation_jobs(self):
        """Test build_precedence_edge_index with single operation jobs."""
        # Create instance where each job has only one operation
        instance = [
            [(0, 5)],  # Job 0: one operation on machine 0, duration 5
            [(1, 3)],  # Job 1: one operation on machine 1, duration 3
            [(2, 4)],  # Job 2: one operation on machine 2, duration 4
        ]

        schedule = Schedule(instance)
        edge_index = schedule.build_precedence_edge_index()

        # Should return empty edge index with proper shape
        assert isinstance(edge_index, np.ndarray)
        assert edge_index.dtype == np.int64
        assert edge_index.shape == (2, 0)

    def test_build_precedence_edge_index_mixed_job_lengths(self):
        """Test build_precedence_edge_index with jobs of different lengths."""
        # Create instance with jobs of different lengths
        instance = [
            [(0, 5), (1, 3)],  # Job 0: 2 operations (1 edge)
            [(2, 4)],  # Job 1: 1 operation (0 edges)
            [(1, 2), (0, 6), (2, 3)],  # Job 2: 3 operations (2 edges)
        ]

        schedule = Schedule(instance)
        edge_index = schedule.build_precedence_edge_index()

        # Should have 1 + 0 + 2 = 3 edges total
        assert edge_index.shape == (2, 3)

        # Create expected node mapping
        op_node_id = {}
        node_id = 0
        for job_idx, job in enumerate(instance):
            for op_idx in range(len(job)):
                op_node_id[(job_idx, op_idx)] = node_id
                node_id += 1

        # Verify specific edges
        sources, targets = edge_index[0], edge_index[1]
        edge_set = set(zip(sources, targets, strict=False))

        expected_edges = {
            (op_node_id[(0, 0)], op_node_id[(0, 1)]),  # Job 0: op0 -> op1
            (op_node_id[(2, 0)], op_node_id[(2, 1)]),  # Job 2: op0 -> op1
            (op_node_id[(2, 1)], op_node_id[(2, 2)]),  # Job 2: op1 -> op2
        }

        assert edge_set == expected_edges

    def test_build_precedence_edge_index_consistency(self, schedule_3x3):
        """Test that edge index is consistent across multiple calls."""
        schedule = schedule_3x3

        edge_index1 = schedule.build_precedence_edge_index()
        edge_index2 = schedule.build_precedence_edge_index()

        # Should return identical results
        assert np.array_equal(edge_index1, edge_index2)

    def test_build_precedence_edge_index_node_ordering(self):
        """Test that node IDs are assigned in the correct order."""
        instance = [
            [(0, 1), (1, 2)],  # Job 0: nodes 0, 1
            [(2, 3)],  # Job 1: node 2
            [(1, 4), (0, 5)],  # Job 2: nodes 3, 4
        ]

        schedule = Schedule(instance)
        edge_index = schedule.build_precedence_edge_index()

        # Expected edges based on sequential node assignment:
        # Job 0: 0 -> 1
        # Job 1: no edges (single operation)
        # Job 2: 3 -> 4
        expected_edges = np.array(
            [[0, 3], [1, 4]],
            dtype=np.int64,  # sources  # targets
        )
        assert np.array_equal(edge_index, expected_edges)

        # Test both directed edges
        edge_index = schedule.build_precedence_edge_index(both_directions=True)
        expected_edges = np.array(
            [[0, 1], [1, 0], [3, 4], [4, 3]],
            dtype=np.int64,  # sources  # targets
        ).T

        assert np.array_equal(edge_index, expected_edges)

        # Test self-loops
        edge_index = schedule.build_precedence_edge_index(
            self_loop=True, both_directions=True
        )
        expected_edges = np.array(
            [[0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [0, 1], [1, 0], [3, 4], [4, 3]],
            dtype=np.int64,  # sources  # targets
        ).T
        assert np.array_equal(
            np.unique(edge_index.T, axis=0), np.unique(expected_edges.T, axis=0)
        )

    def test_build_precedence_edge_index_with_self_loops(self, schedule_3x3):
        instance = schedule_3x3.instance
        edge_index = schedule_3x3.build_precedence_edge_index(self_loop=True)
        # Create the node mapping to verify edges
        op_node_id = {}
        node_id = 0
        for job_idx, job in enumerate(instance):
            for op_idx in range(len(job)):
                op_node_id[(job_idx, op_idx)] = node_id
                node_id += 1
        # Verify that edges represent correct precedence constraints
        sources, targets = edge_index[0], edge_index[1]
        edge_set = set(zip(sources, targets, strict=False))
        expected_edges = set()
        for job_idx, job in enumerate(instance):
            for op_idx in range(len(job)):
                source_node = op_node_id[(job_idx, op_idx)]
                target_node = op_node_id[(job_idx, op_idx)]
                expected_edges.add((source_node, target_node))
            for op_idx in range(len(job) - 1):
                source_node = op_node_id[(job_idx, op_idx)]
                target_node = op_node_id[(job_idx, op_idx + 1)]
                expected_edges.add((source_node, target_node))
        assert edge_set == expected_edges

    def test_build_machine_edge_index_basic(self, schedule_3x3):
        edge_index = schedule_3x3.build_machine_edge_index()

        # Check return type and shape
        assert isinstance(edge_index, np.ndarray)

        assert edge_index.dtype == np.int64
        assert edge_index.shape[0] == 2  # Should have 2 rows

        all_edges = np.array(
            [
                [0, 3],
                [3, 0],
                [0, 8],
                [8, 0],
                [3, 8],
                [8, 3],
                [1, 5],
                [5, 1],
                [1, 6],
                [6, 1],
                [5, 6],
                [6, 5],
                [2, 4],
                [4, 2],
                [2, 7],
                [7, 2],
                [4, 7],
                [7, 4],
            ]
        )

        edge_list = [tuple(e) for e in edge_index.T.tolist()]
        for edge in all_edges:
            assert tuple(edge) in edge_list
