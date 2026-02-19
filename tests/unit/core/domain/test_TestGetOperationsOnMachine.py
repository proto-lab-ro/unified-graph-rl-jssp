from jssp_core.schedule.jssp import Schedule


class TestGetOperationsOnMachine:
    """Tests for Schedule.get_operations_on_machine method"""

    def test_get_operations_on_single_machine(self):
        """Test getting operations for a machine with multiple operations"""
        # Instance where machine 0 has multiple operations
        instance = [
            [(0, 10), (1, 20)],  # Job 0: op0 on machine 0, op1 on machine 1
            [(0, 15), (2, 25)],  # Job 1: op0 on machine 0, op1 on machine 2
        ]
        schedule = Schedule(instance)

        ops = schedule.get_operations_on_machine(0)

        assert len(ops) == 2
        assert (0, 0) in ops
        assert (1, 0) in ops

    def test_get_operations_on_machine_all_operations(self):
        """Test machine that has all operations"""
        instance = [
            [(0, 10)],
            [(0, 15)],
            [(0, 20)],
        ]
        schedule = Schedule(instance)

        ops = schedule.get_operations_on_machine(0)

        assert len(ops) == 3
        assert (0, 0) in ops
        assert (1, 0) in ops
        assert (2, 0) in ops

    def test_get_operations_preserves_order(self):
        """Test that operations are returned in job/operation order"""
        instance = [
            [(1, 10), (0, 20)],  # Job 0
            [(0, 15), (1, 25)],  # Job 1
            [(1, 30)],  # Job 2
        ]
        schedule = Schedule(instance)

        ops = schedule.get_operations_on_machine(1)

        # Should be in order: (job0, op0), (job1, op1), (job2, op0)
        assert ops == [(0, 0), (1, 1), (2, 0)]

    def test_get_operations_multiple_machines(self):
        """Test getting operations from different machines"""
        instance = [
            [(0, 10), (1, 20), (2, 30)],
            [(1, 15), (2, 25), (0, 35)],
        ]
        schedule = Schedule(instance)

        ops_m0 = schedule.get_operations_on_machine(0)
        ops_m1 = schedule.get_operations_on_machine(1)
        ops_m2 = schedule.get_operations_on_machine(2)

        assert ops_m0 == [(0, 0), (1, 2)]
        assert ops_m1 == [(0, 1), (1, 0)]
        assert ops_m2 == [(0, 2), (1, 1)]

    def test_get_operations_single_job_multiple_ops_same_machine(self):
        """Test single job with multiple operations on same machine"""
        instance = [
            [(0, 10), (1, 20), (0, 30)],  # Job uses machine 0 twice
        ]
        schedule = Schedule(instance)

        ops = schedule.get_operations_on_machine(0)

        assert len(ops) == 2
        assert (0, 0) in ops
        assert (0, 2) in ops
