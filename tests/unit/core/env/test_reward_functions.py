"""
Unit tests for JSSP reward functions.
"""

import numpy as np
import pytest

from jssp_core.instances import F3X3_INSTANCE, _parse_instance
from jssp_core.reward_functions import (
    DenseShapedReward,
    LowerBoundMakespanReward,
    MakespanImprovementReward,
    MaxOperationLowerBoundDifferenceReward,
    OperationLowerBoundSpreadReward,
    Q90GapNormalizedReward,
    SparseExponentialReward,
    SparseMakespanReward,
    get_reward_function,
    list_reward_functions,
)
from jssp_core.schedule.jssp import Schedule


class DummySchedule:
    """Lightweight schedule stub for deterministic reward calculations."""

    def __init__(
        self,
        est_completion: float = 0.0,
        makespan: float = 0.0,
        lower_bound: float = 0.0,
        maximal_bound: float = 100.0,
        scheduled_ops: int = 0,
        total_ops: int = 1,
        complete: bool = False,
        operation_bounds: dict[tuple[int, int], float] | None = None,
        eligible_jobs: list[int] | None = None,
    ):
        self._est_completion = est_completion
        self._makespan = makespan
        self._lower_bound = lower_bound
        self._maximal_bound = maximal_bound
        self._scheduled_ops = scheduled_ops
        self.num_operations = total_ops
        self._complete = complete
        self._operation_bounds = operation_bounds or {}
        self._eligible_jobs = eligible_jobs or []

    def estimate_completion_time(self, heuristic: str = "MWKR") -> float:
        return self._est_completion

    def get_makespan(self) -> float:
        return self._makespan

    def get_scheduled_operations_count(self) -> int:
        return self._scheduled_ops

    def get_lower_bound_makespan(self) -> float:
        return self._lower_bound

    def get_maximal_bound_makespan(self) -> float:
        return self._maximal_bound

    def is_complete(self) -> bool:
        return self._complete

    def get_operation_lower_bounds(self) -> dict[tuple[int, int], float]:
        return self._operation_bounds

    def get_eligible_jobs(self) -> list[int]:
        return self._eligible_jobs


@pytest.mark.unit
class TestMaxOperationLowerBoundDifferenceReward:
    """Test MaxOperationLowerBoundDifferenceReward class."""

    @pytest.fixture
    def simple_instance(self):
        """Create a simple 3x3 instance for testing."""
        return _parse_instance(F3X3_INSTANCE)

    @pytest.fixture
    def initial_schedule(self, simple_instance):
        """Create an initial schedule for testing."""
        return Schedule(simple_instance)

    @pytest.fixture
    def reward_function(self, initial_schedule):
        """Create a MaxOperationLowerBoundDifferenceReward instance."""
        return MaxOperationLowerBoundDifferenceReward(initial_schedule=initial_schedule)

    def test_initialization(self, initial_schedule):
        """Test that reward function initializes correctly."""
        reward_fn = MaxOperationLowerBoundDifferenceReward(
            initial_schedule=initial_schedule
        )
        assert reward_fn.initial_schedule == initial_schedule

    def test_name_property(self, reward_function):
        """Test that name property returns correct value."""
        assert reward_function.name == "max_operation_lower_bound_difference"

    def test_calculate_reward_with_improvement(self, reward_function, simple_instance):
        """Test reward calculation when lower bounds improve (decrease)."""
        # Create two schedules representing before and after states
        schedule_before = Schedule(simple_instance)
        schedule_after = Schedule(simple_instance)

        # Schedule an operation in the after state to create improvement
        # This should reduce the lower bounds for remaining operations
        schedule_after.schedule_job(0)  # Schedule first job's first operation

        reward_data = {
            "state_before": schedule_before,
            "state_after": schedule_after,
            "schedule": schedule_after,
            "is_complete": False,
        }

        reward = reward_function.calculate_reward(reward_data)

        # Reward should be a float
        assert isinstance(reward, (float, np.floating))

        # Schedule one first operation should not lead to a change in max lower bound difference
        assert reward == 0

    def test_calculate_reward_no_improvement(self, reward_function, simple_instance):
        """Test reward calculation when there's no improvement."""
        # Create identical schedules (no action taken)
        schedule_before = Schedule(simple_instance)
        schedule_after = Schedule(simple_instance)

        reward_data = {
            "state_before": schedule_before,
            "state_after": schedule_after,
            "schedule": schedule_after,
            "is_complete": False,
        }

        reward = reward_function.calculate_reward(reward_data)

        # With no change, the difference should be 0, so reward should be 0
        assert reward == 0.0

    def test_calculate_reward_multiple_steps(self, reward_function, simple_instance):
        """Test reward calculation over multiple scheduling steps."""
        schedule_before = Schedule(simple_instance)

        # Schedule multiple operations
        eligible_jobs = schedule_before.get_eligible_jobs()
        rewards = []

        for _ in range(min(3, len(eligible_jobs))):
            schedule_after = Schedule(simple_instance)
            schedule_after.scheduled = schedule_before.scheduled.copy()
            schedule_after.job_next_op = schedule_before.job_next_op.copy()
            schedule_after.job_ready_time = schedule_before.job_ready_time.copy()
            schedule_after.machine_ready_time = (
                schedule_before.machine_ready_time.copy()
            )
            schedule_after.eligible_operations = (
                schedule_before.eligible_operations.copy()
            )

            eligible = schedule_after.get_eligible_jobs()
            if eligible:
                schedule_after.schedule_job(eligible[0])

                reward_data = {
                    "state_before": schedule_before,
                    "state_after": schedule_after,
                    "schedule": schedule_after,
                    "is_complete": False,
                }

                reward = reward_function.calculate_reward(reward_data)
                rewards.append(reward)

                # Move to next state
                schedule_before = schedule_after

        # Should have collected multiple rewards
        assert len(rewards) > 0
        # All rewards should be numeric
        assert all(isinstance(r, (float, np.floating)) for r in rewards)

    def test_with_completed_schedule(self, reward_function, simple_instance):
        """Test reward calculation when schedule is complete."""
        schedule_before = Schedule(simple_instance)

        # Schedule all operations
        while not schedule_before.is_complete():
            eligible = schedule_before.get_eligible_jobs()
            if eligible:
                schedule_before.schedule_job(eligible[0])
            else:
                break

        # Create an after state (which should also be complete)
        schedule_after = schedule_before

        reward_data = {
            "state_before": schedule_before,
            "state_after": schedule_after,
            "schedule": schedule_after,
            "is_complete": True,
        }

        # Should still calculate reward without errors
        reward = reward_function.calculate_reward(reward_data)
        assert isinstance(reward, (float, np.floating))

    def test_factory_function(self, initial_schedule):
        """Test that factory function correctly creates the reward function."""
        reward_fn = get_reward_function(
            name="max_operation_lower_bound_difference",
            initial_schedule=initial_schedule,
        )

        assert isinstance(reward_fn, MaxOperationLowerBoundDifferenceReward)
        assert reward_fn.lb == initial_schedule.get_lower_bound_makespan()

    def test_listed_in_available_functions(self):
        """Test that the reward function is listed in available functions."""
        available_functions = list_reward_functions()

        assert "max_operation_lower_bound_difference" in available_functions
        assert (
            available_functions["max_operation_lower_bound_difference"]
            == "max_operation_lower_bound_difference"
        )

    def test_reward_deterministic(self, reward_function, simple_instance):
        """Test that reward calculation is deterministic for same inputs."""
        schedule_before = Schedule(simple_instance)
        schedule_after = Schedule(simple_instance)
        schedule_after.schedule_job(0)

        reward_data = {
            "state_before": schedule_before,
            "state_after": schedule_after,
            "schedule": schedule_after,
            "is_complete": False,
        }

        # Calculate reward multiple times
        reward1 = reward_function.calculate_reward(reward_data)
        reward2 = reward_function.calculate_reward(reward_data)
        reward3 = reward_function.calculate_reward(reward_data)

        # All should be identical
        assert reward1 == reward2 == reward3


@pytest.mark.unit
class TestRewardFunctionCalculations:
    """Unit tests for individual reward function calculations."""

    def test_makespan_improvement_reward_scaling_and_offset(self):
        initial = DummySchedule(est_completion=20.0)
        reward_fn = MakespanImprovementReward(
            initial_schedule=initial, completion_bonus=7.5, offset=0.1
        )

        state_before = DummySchedule(est_completion=15.0)
        state_after = DummySchedule(est_completion=10.0)

        reward = reward_fn.calculate_reward(
            {
                "state_before": state_before,
                "state_after": state_after,
                "is_complete": False,
            }
        )

        # 10 * (offset + improvement / initial_est_makespan) = 10 * (0.1 + 0.25) = 3.5
        assert reward == pytest.approx(3.5)

    def test_makespan_improvement_reward_completion_bonus(self):
        initial = DummySchedule(est_completion=25.0)
        reward_fn = MakespanImprovementReward(
            initial_schedule=initial, completion_bonus=4.2
        )

        reward = reward_fn.calculate_reward(
            {
                "state_before": DummySchedule(est_completion=20.0),
                "state_after": DummySchedule(est_completion=5.0),
                "is_complete": True,
            }
        )

        assert reward == 4.2

    def test_dense_shaped_reward_components(self):
        initial = DummySchedule(est_completion=20.0, total_ops=10)
        reward_fn = DenseShapedReward(
            initial_schedule=initial,
            makespan_weight=1.0,
            progress_weight=0.1,
            efficiency_weight=0.5,
            completion_bonus=2.0,
        )

        schedule_before = DummySchedule(makespan=30.0)
        schedule_after = DummySchedule(makespan=20.0, scheduled_ops=4, total_ops=10)

        reward = reward_fn.calculate_reward(
            {
                "state_before": schedule_before,
                "state_after": schedule_after,
                "schedule": schedule_after,
                "is_complete": False,
            }
        )

        # makespan_reward = (30-20)/20 = 0.5
        # progress_reward = 0.1 * (4/10) = 0.04
        # efficiency term clamps to 0 because 20/(4*20/10)=2.5 > 1
        assert reward == pytest.approx(0.54)

    def test_dense_shaped_reward_completion_bonus_and_efficiency(self):
        initial = DummySchedule(est_completion=10.0, total_ops=4)
        reward_fn = DenseShapedReward(
            initial_schedule=initial,
            makespan_weight=1.0,
            progress_weight=0.2,
            efficiency_weight=0.5,
            completion_bonus=3.0,
        )

        schedule_before = DummySchedule(makespan=8.0)
        schedule_after = DummySchedule(
            makespan=4.0, scheduled_ops=4, total_ops=4, complete=True
        )

        reward = reward_fn.calculate_reward(
            {
                "state_before": schedule_before,
                "state_after": schedule_after,
                "schedule": schedule_after,
                "is_complete": True,
            }
        )

        # Efficiency is positive here: 1 - 4/(4*10/4)=0.6, weighted to 0.3
        # makespan_reward = (8-4)/10 = 0.4, progress_reward = 0.2 * 1 = 0.2
        assert reward == pytest.approx(0.9 + 3.0)

    def test_lower_bound_makespan_reward_difference(self):
        reward_fn = LowerBoundMakespanReward(initial_schedule=DummySchedule())

        reward = reward_fn.calculate_reward(
            {
                "state_before": DummySchedule(lower_bound=15.0),
                "state_after": DummySchedule(lower_bound=10.0),
            }
        )

        assert reward == 5.0

    def test_sparse_makespan_reward_complete_vs_incomplete(self):
        initial = DummySchedule(lower_bound=12.0)
        reward_fn = SparseMakespanReward(initial_schedule=initial)

        incomplete_reward = reward_fn.calculate_reward(
            {
                "state_after": DummySchedule(makespan=20.0, complete=False),
                "schedule": DummySchedule(makespan=20.0, complete=False),
            }
        )
        complete_reward = reward_fn.calculate_reward(
            {
                "state_after": DummySchedule(makespan=24.0, complete=True),
                "schedule": DummySchedule(makespan=24.0, complete=True),
            }
        )

        assert incomplete_reward == 0.0
        assert complete_reward == pytest.approx(0.5)

    def test_sparse_exponential_reward(self):
        initial = DummySchedule(lower_bound=10.0)
        reward_fn = SparseExponentialReward(initial_schedule=initial, scaling_value=5.0)

        incomplete = reward_fn.calculate_reward(
            {"schedule": DummySchedule(makespan=12.0), "is_complete": False}
        )
        complete = reward_fn.calculate_reward(
            {"schedule": DummySchedule(makespan=15.0), "is_complete": True}
        )

        assert incomplete == 0
        assert complete == pytest.approx(np.exp(-1.0))

    def test_operation_lower_bound_spread_reward(self):
        reward_fn = OperationLowerBoundSpreadReward(initial_schedule=DummySchedule())

        before_bounds = {(0, 0): 10.0, (0, 1): 20.0, (1, 0): 30.0}
        after_bounds = {(0, 0): 12.0, (0, 1): 22.0, (1, 0): 26.0}

        reward = reward_fn.calculate_reward(
            {
                "state_before": DummySchedule(operation_bounds=before_bounds),
                "state_after": DummySchedule(operation_bounds=after_bounds),
            }
        )

        # Spread before: 30 - (10+20+30)/3 = 30 - 20 = 10
        # Spread after: 26 - (12+22+26)/3 = 26 - 20 = 6
        # Reward: (10 - 6) * 0.002 = 0.008
        assert reward == pytest.approx(0.008)

    def test_negative_sparse_makespan_reward(self):
        initial = DummySchedule(lower_bound=100.0)
        reward_fn = get_reward_function(
            "negative_sparse_makespan", initial, scaling_factor=2
        )

        incomplete = reward_fn.calculate_reward(
            {"schedule": DummySchedule(makespan=120.0, complete=False)}
        )
        complete = reward_fn.calculate_reward(
            {"schedule": DummySchedule(makespan=120.0, complete=True)}
        )

        assert incomplete == 0.0
        # Reward = -makespan / (lb * scaling_factor) = -120 / (100 * 2) = -0.6
        assert complete == pytest.approx(-0.6)

    def test_bounded_negative_sparse_makespan_reward(self):
        initial = DummySchedule(lower_bound=100.0, maximal_bound=200.0)
        reward_fn = get_reward_function("bounded_negative_sparse_makespan", initial)

        complete = reward_fn.calculate_reward(
            {"schedule": DummySchedule(makespan=150.0, complete=True)}
        )
        # Reward = -((makespan - lb) / (mb - lb)) = -((150-100) / (200-100)) = -0.5
        assert complete == pytest.approx(-0.5)

    def test_sparse_lb2_makespan_reward(self):
        initial = DummySchedule(lower_bound=100.0, maximal_bound=200.0)
        reward_fn = get_reward_function("sparse_lb2_makespan", initial)

        complete = reward_fn.calculate_reward(
            {"schedule": DummySchedule(makespan=120.0, complete=True)}
        )
        # Reward = (lb - makespan) / (mb - lb) = (100 - 120) / (200 - 100) = -0.2
        assert complete == pytest.approx(-0.2)

    def test_sparse_normalized_bound_scale_reward(self):
        initial = DummySchedule(lower_bound=100.0, maximal_bound=200.0)
        reward_fn = get_reward_function("sparse_normalized_bound_scale", initial)

        complete = reward_fn.calculate_reward(
            {"schedule": DummySchedule(makespan=120.0, complete=True)}
        )
        # Reward = (mb - makespan) / (mb - lb) = (200 - 120) / (200 - 100) = 0.8
        assert complete == pytest.approx(0.8)

    def test_operation_lower_bound_spread_normalized_reward(self):
        initial = DummySchedule(lower_bound=50.0, maximal_bound=100.0)

        # Test initial_lb normalization
        reward_fn = get_reward_function(
            "operation_lower_bound_spread_normalized",
            initial,
            normalization="initial_lb",
            scaling_factor=1.0,
        )

        before_bounds = {(0, 0): 10.0, (0, 1): 30.0}  # max=30, mean=20, spread=10
        after_bounds = {(0, 0): 12.0, (0, 1): 28.0}  # max=28, mean=20, spread=8

        reward = reward_fn.calculate_reward(
            {
                "state_before": DummySchedule(operation_bounds=before_bounds),
                "state_after": DummySchedule(operation_bounds=after_bounds),
            }
        )
        # (10 - 8) / 50 = 0.04
        assert reward == pytest.approx(0.04)

        # Test current_max normalization
        reward_fn_curr = get_reward_function(
            "operation_lower_bound_spread_normalized",
            initial,
            normalization="current_max",
        )
        reward = reward_fn_curr.calculate_reward(
            {
                "state_before": DummySchedule(operation_bounds=before_bounds),
                "state_after": DummySchedule(operation_bounds=after_bounds),
            }
        )
        # spread_before/max_before - spread_after/max_after = 10/30 - 8/28 = 1/3 - 2/7 = (7-6)/21 = 1/21
        assert reward == pytest.approx(1 / 21)

    def test_q90_gap_normalized_reward(self):
        initial = DummySchedule(lower_bound=100.0)
        reward_fn = get_reward_function("q90_gap_normalized", initial, quantile=50.0)

        # Use median for easier calculation
        before_bounds = {i: float(i) for i in range(1, 11)}  # 1..10, max=10, median=5.5
        after_bounds = {
            i: float(i + 1) for i in range(1, 11)
        }  # 2..11, max=11, median=6.5

        # gap = (max - q) / max
        # gap_before = (10 - 5.5) / 10 = 0.45
        # gap_after = (11 - 6.5) / 11 = 4.5 / 11 approx 0.409
        reward = reward_fn.calculate_reward(
            {
                "state_before": DummySchedule(operation_bounds=before_bounds),
                "state_after": DummySchedule(operation_bounds=after_bounds),
            }
        )
        assert reward == pytest.approx(0.45 - 4.5 / 11)

    def test_fraction_near_critical_reward(self):
        initial = DummySchedule(lower_bound=100.0)
        reward_fn = get_reward_function(
            "fraction_near_critical", initial, threshold=0.9
        )

        before_bounds = {
            (0, 0): 100.0,
            (0, 1): 80.0,
        }  # max=100, threshold*max=90. Near critical: 1/2 = 0.5
        after_bounds = {
            (0, 0): 100.0,
            (0, 1): 95.0,
        }  # max=100, threshold*max=90. Near critical: 2/2 = 1.0

        reward = reward_fn.calculate_reward(
            {
                "state_before": DummySchedule(operation_bounds=before_bounds),
                "state_after": DummySchedule(operation_bounds=after_bounds),
            }
        )
        assert reward == pytest.approx(0.5)

    def test_waiting_jobs_reward(self):
        reward_fn = get_reward_function("waiting_jobs", DummySchedule())

        reward = reward_fn.calculate_reward(
            {"state_before": DummySchedule(eligible_jobs=[1, 2, 3])}
        )
        assert reward == -3.0

    def test_reward_mixer(self):
        initial = DummySchedule(lower_bound=10.0)
        reward_fn = get_reward_function(
            "reward_mixer",
            initial,
            reward_functions=["lower_bound_makespan", "waiting_jobs"],
            weights=[2.0, 0.5],
        )

        reward = reward_fn.calculate_reward(
            {
                "state_before": DummySchedule(lower_bound=15.0, eligible_jobs=[1, 2]),
                "state_after": DummySchedule(lower_bound=10.0),
            }
        )

        # 2.0 * (15-10) + 0.5 * (-2) = 10 - 1 = 9.0
        assert reward == pytest.approx(9.0)

    def test_reward_mixer_with_specific_kwargs(self):
        initial = DummySchedule(lower_bound=10.0)
        reward_fn = get_reward_function(
            "reward_mixer",
            initial,
            reward_functions=["sparse_exponential", "waiting_jobs"],
            reward_specific_kwargs={"sparse_exponential": {"scaling_value": 10.0}},
        )

        # Check that scaling_value was passed
        exp_reward = next(
            r for r in reward_fn.rewards if r.name == "sparse_exponential"
        )
        assert exp_reward.scaling_value == 10.0

    def test_reward_mixer_global_kwargs(self):
        initial = DummySchedule(lower_bound=10.0)
        # Use a reward that accepts scaling_value, and pass it globally
        reward_fn = get_reward_function(
            "reward_mixer",
            initial,
            reward_functions=["sparse_exponential"],
            scaling_value=42.0,
        )

        exp_reward = reward_fn.rewards[0]
        assert exp_reward.scaling_value == 42.0

    def test_edge_case_empty_bounds(self):
        """Test that rewards based on bounds don't crash with empty data."""
        reward_fn = OperationLowerBoundSpreadReward(initial_schedule=DummySchedule())
        reward = reward_fn.calculate_reward(
            {
                "state_before": DummySchedule(operation_bounds={}),
                "state_after": DummySchedule(operation_bounds={}),
            }
        )
        assert reward == 0.0

        reward_fn_q = Q90GapNormalizedReward(initial_schedule=DummySchedule())
        reward_q = reward_fn_q.calculate_reward(
            {
                "state_before": DummySchedule(operation_bounds={}),
                "state_after": DummySchedule(operation_bounds={}),
            }
        )
        assert reward_q == 0.0


@pytest.mark.unit
class TestRewardFunctionRegistry:
    """Test reward function registry and factory functions."""

    def test_list_reward_functions_returns_dict(self):
        """Test that list_reward_functions returns a dictionary."""
        functions = list_reward_functions()
        assert isinstance(functions, dict)
        assert len(functions) > 0

    def test_get_reward_function_unknown_name(self, schedule_3x3):
        """Test that unknown reward function name raises ValueError."""
        with pytest.raises(ValueError, match="Unknown component"):
            get_reward_function(
                name="nonexistent_reward", initial_schedule=schedule_3x3
            )

    def test_get_reward_function_with_kwargs(self, schedule_3x3):
        """Test that kwargs are passed to reward function constructor."""
        reward_fn = get_reward_function(
            name="max_operation_lower_bound_difference",
            initial_schedule=schedule_3x3,
        )

        assert reward_fn.lb == schedule_3x3.get_lower_bound_makespan()
