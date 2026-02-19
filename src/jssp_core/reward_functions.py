"""
Reward functions for JSSP environment.
Allows easy experimentation with different reward strategies.
"""

from typing import Any

import numpy as np

from jssp_core.domain.reward import RewardFunction
from jssp_core.registry import REWARD_REGISTRY
from jssp_core.schedule import Schedule


@REWARD_REGISTRY.register("makespan_improvement")
class MakespanImprovementReward(RewardFunction):
    """Original reward function based on makespan improvement"""

    def __init__(
        self,
        initial_schedule: Schedule,
        heuristic: str = "MWKR",
        completion_bonus: float = 1.0,
        offset: float = 0.0,
        **kwargs,
    ):
        """
        Initialize the reward function with the given parameters.
        """
        self.initial_schedule = initial_schedule
        self.completion_bonus = completion_bonus
        self.heuristic = heuristic
        self.offset = offset
        self.initial_est_makespan = self.initial_schedule.estimate_completion_time(
            heuristic=self.heuristic
        )

    def calculate_reward(self, reward_data: dict[str, Any]) -> float:
        schedule_before: Schedule = reward_data["state_before"]
        schedule_after: Schedule = reward_data["state_after"]
        is_complete = reward_data["is_complete"]

        reward = 10 * (
            self.offset
            + (
                (
                    schedule_before.estimate_completion_time(heuristic=self.heuristic)
                    - schedule_after.estimate_completion_time(heuristic=self.heuristic)
                )
                / self.initial_est_makespan
            )
        )

        if is_complete:
            reward = self.completion_bonus

        return reward

    @property
    def name(self) -> str:
        return "makespan_improvement"


@REWARD_REGISTRY.register("dense_shaped")
class DenseShapedReward(RewardFunction):
    """Dense reward with multiple shaping factors"""

    def __init__(
        self,
        initial_schedule: Schedule,
        heuristic: str = "MWKR",
        makespan_weight: float = 1.0,
        progress_weight: float = 0.1,
        completion_bonus: float = 5.0,
        efficiency_weight: float = 0.5,
        **kwargs,
    ):
        super().__init__(initial_schedule, **kwargs)
        self.makespan_weight = makespan_weight
        self.progress_weight = progress_weight
        self.completion_bonus = completion_bonus
        self.efficiency_weight = efficiency_weight
        self.heuristic = heuristic
        self.initial_est_makespan = self.initial_schedule.estimate_completion_time(
            heuristic=self.heuristic
        )

    def calculate_reward(self, reward_data: dict[str, Any]) -> float:
        schedule_before: Schedule = reward_data["state_before"]
        schedule_after: Schedule = reward_data["state_after"]
        is_complete = reward_data["is_complete"]
        schedule = reward_data["schedule"]

        # Makespan improvement component
        makespan_improvement = (
            schedule_before.get_makespan() - schedule_after.get_makespan()
        ) / self.initial_est_makespan
        makespan_reward = self.makespan_weight * makespan_improvement

        # Progress component (reward for scheduling operations)
        total_ops = schedule.num_operations
        scheduled_ops = schedule.get_scheduled_operations_count()
        progress_reward = self.progress_weight * (scheduled_ops / total_ops)

        # Machine utilization efficiency
        if scheduled_ops > 0:
            efficiency = 1.0 - (
                schedule_after.get_makespan()
                / (scheduled_ops * self.initial_est_makespan / total_ops)
            )
            efficiency_reward = self.efficiency_weight * max(0, efficiency)
        else:
            efficiency_reward = 0

        reward = makespan_reward + progress_reward + efficiency_reward

        if is_complete:
            reward += self.completion_bonus

        return reward

    @property
    def name(self) -> str:
        return "dense_shaped"


@REWARD_REGISTRY.register("lower_bound_makespan")
class LowerBoundMakespanReward(RewardFunction):
    """
    Reward function based on the difference in lower bound makespan estimates
    between consecutive states.
    R(at, st) = H(st) - H(st+1), where H(st) = max_{i,j} CLB(Oij, st)
    """

    def __init__(self, initial_schedule: Schedule, **kwargs):
        super().__init__(initial_schedule, **kwargs)

    def calculate_reward(self, reward_data: dict[str, Any]) -> float:
        schedule_before: Schedule = reward_data["state_before"]
        schedule_after: Schedule = reward_data["state_after"]

        # H(st): lower bound makespan for state_before
        lb_before = schedule_before.get_lower_bound_makespan()
        # H(st+1): lower bound makespan for state_after
        lb_after = schedule_after.get_lower_bound_makespan()

        reward = lb_before - lb_after
        return reward

    @property
    def name(self) -> str:
        return "lower_bound_makespan"


@REWARD_REGISTRY.register("sparse_makespan")
class SparseMakespanReward(RewardFunction):
    """
    Sparse reward function that only provides a reward at the end of the episode
    based on the makespan of the final schedule.
    """

    def __init__(self, initial_schedule: Schedule, **kwargs):
        super().__init__(initial_schedule, **kwargs)

        self.lb = initial_schedule.get_lower_bound_makespan()

    def calculate_reward(self, reward_data: dict[str, Any]) -> float:
        schedule: Schedule = reward_data["schedule"]
        reward = 0.0
        if schedule.is_complete():
            reward = self.lb / schedule.get_makespan()
        return reward

    @property
    def name(self) -> str:
        return "sparse_makespan"


@REWARD_REGISTRY.register("negative_sparse_makespan")
class NegativeSparseMakespanReward(RewardFunction):
    """
    Sparse reward function that only provides a reward at the end of the episode
    based on the makespan of the final schedule.
    ️   R = - (makespan / lower_bound)
    ️   A lower makespan results in a less negative reward.
    ️   scaling_factor scales the magnitude of the reward.
    """

    def __init__(
        self, initial_schedule: Schedule, scaling_factor: float = 100, **kwargs
    ):
        super().__init__(initial_schedule, **kwargs)

        self.lb = initial_schedule.get_lower_bound_makespan()
        self.scaling_factor = scaling_factor

    def calculate_reward(self, reward_data: dict[str, Any]) -> float:
        schedule: Schedule = reward_data["schedule"]
        reward = 0.0
        if schedule.is_complete():
            reward = -schedule.get_makespan() / (self.lb * self.scaling_factor)
        return reward

    @property
    def name(self) -> str:
        return "negative_sparse_makespan"


@REWARD_REGISTRY.register("bounded_negative_sparse_makespan")
class BoundedNegativeSparseMakespanReward(RewardFunction):
    """
    Sparse reward function that only provides a reward at the end of the episode
    based on the makespan of the final schedule.
    """

    def __init__(self, initial_schedule: Schedule, **kwargs):
        super().__init__(initial_schedule, **kwargs)
        self.mb = initial_schedule.get_maximal_bound_makespan()
        self.lb = initial_schedule.get_lower_bound_makespan()

    def calculate_reward(self, reward_data: dict[str, Any]) -> float:
        schedule: Schedule = reward_data["schedule"]
        reward = 0.0
        if schedule.is_complete():
            reward = -((schedule.get_makespan() - self.lb) / (self.mb - self.lb))
        return reward

    @property
    def name(self) -> str:
        return "bounded_negative_sparse_makespan"


@REWARD_REGISTRY.register("sparse_lb2_makespan")
class SparseLb2MakespanReward(RewardFunction):
    """
    Sparse reward function that only provides a reward at the end of the episode
    based on the makespan of the final schedule.
    """

    def __init__(self, initial_schedule: Schedule, **kwargs):
        super().__init__(initial_schedule, **kwargs)

        self.lb = initial_schedule.get_lower_bound_makespan()
        self.mb = initial_schedule.get_maximal_bound_makespan()

    def calculate_reward(self, reward_data: dict[str, Any]) -> float:
        schedule: Schedule = reward_data["schedule"]
        reward = 0.0
        if schedule.is_complete():
            reward = (self.lb - schedule.get_makespan()) / (self.mb - self.lb)
        return reward

    @property
    def name(self) -> str:
        return "sparse_lb2_makespan"


@REWARD_REGISTRY.register("sparse_normalized_bound_scale")
class SparseNormalizedBoundScaleReward(RewardFunction):
    """
    Sparse reward function that returns 0 for non-terminal states and
    returns normalized bound scale at completion.
    Normalized Bound Scale = (LB - Makespan) / LB
    """

    def __init__(self, initial_schedule: Schedule, **kwargs):
        super().__init__(initial_schedule, **kwargs)
        self.lb = initial_schedule.get_lower_bound_makespan()
        self.mb = initial_schedule.get_maximal_bound_makespan()

    def calculate_reward(self, reward_data: dict[str, Any]) -> float:
        schedule = reward_data["schedule"]

        if schedule.is_complete():
            makespan = schedule.get_makespan()
            reward = (self.mb - makespan) / (self.mb - self.lb)
            return reward
        return 0

    @property
    def name(self) -> str:
        return "sparse_normalized_bound_scale"


@REWARD_REGISTRY.register("sparse_exponential")
class SparseExponentialReward(RewardFunction):
    """
    Sparse reward function that returns 0 for non-terminal states and
    returns negative difference from lower bound (55) at completion.
    A larger difference from lower bound results in a more negative reward.
    """

    def __init__(self, initial_schedule: Schedule, scaling_value: float = 20, **kwargs):
        super().__init__(initial_schedule, **kwargs)
        self.lb = initial_schedule.get_lower_bound_makespan()
        self.scaling_value = scaling_value

    def calculate_reward(self, reward_data: dict[str, Any]) -> float:
        is_complete = reward_data["is_complete"]
        schedule = reward_data["schedule"]

        if is_complete:
            makespan = schedule.get_makespan()
            reward = self._reward_function(makespan)
            return reward
        return 0

    def _reward_function(self, makespan: float) -> float:
        """Exponential reward"""
        return np.exp(-(makespan - self.lb) / self.scaling_value)

    @property
    def name(self) -> str:
        return "sparse_exponential"


@REWARD_REGISTRY.register("max_operation_lower_bound_difference")
class MaxOperationLowerBoundDifferenceReward(RewardFunction):
    """
    Reward based on the maximum operation lower bound difference
    between consecutive states.
    R(at, st) = max_{i,j} (CLB(Oij, st) - CLB(Oij, st+1)) * scaling_factor
    """

    def __init__(self, initial_schedule: Schedule, **kwargs):
        super().__init__(initial_schedule, **kwargs)
        self.lb = initial_schedule.get_lower_bound_makespan()

    def calculate_reward(self, reward_data: dict[str, Any]) -> float:
        clbs_before = reward_data["state_before"].get_operation_lower_bounds()
        clbs_after = reward_data["state_after"].get_operation_lower_bounds()

        # Compute max difference directly without building intermediate structures
        max_col = max(k[1] for k in clbs_before.keys())
        max_diff = min(
            clbs_before[k] - clbs_after[k]
            for k in clbs_before.keys()
            if k[1] == max_col and k in clbs_after
        )

        return max_diff / self.lb

    @property
    def name(self) -> str:
        return "max_operation_lower_bound_difference"


@REWARD_REGISTRY.register("operation_lower_bound_spread")
class OperationLowerBoundSpreadReward(RewardFunction):
    """
    Reward based on the reduction of the spread between the maximum and mean
    operation lower bounds between consecutive states.
    """

    def __init__(
        self,
        initial_schedule: Schedule,
        scaling_factor: float = 0.002,
        **kwargs,
    ):
        super().__init__(initial_schedule, **kwargs)
        self.scaling_factor = scaling_factor

    def calculate_reward(self, reward_data: dict[str, Any]) -> float:
        clbs_before = list(
            reward_data["state_before"].get_operation_lower_bounds().values()
        )
        clbs_after = list(
            reward_data["state_after"].get_operation_lower_bounds().values()
        )

        if not clbs_before or not clbs_after:
            return 0.0

        # Compute spread (max - mean) before and after
        spread_before = max(clbs_before) - sum(clbs_before) / len(clbs_before)
        spread_after = max(clbs_after) - sum(clbs_after) / len(clbs_after)

        # Positive reward if spread decreased (improved balance)
        reward = (spread_before - spread_after) * self.scaling_factor
        return reward

    @property
    def name(self) -> str:
        return "operation_lower_bound_spread"


@REWARD_REGISTRY.register("operation_lower_bound_spread_normalized")
class OperationLowerBoundSpreadRewardNormalized(RewardFunction):
    def __init__(
        self,
        initial_schedule: Schedule,
        scaling_factor: float = 1,
        normalization: str | None = None,
        penalty_for_critical_path_worsening: bool = False,
        **kwargs,
    ):
        super().__init__(initial_schedule, **kwargs)
        self.normalization = normalization
        self.scaling_factor = scaling_factor
        self.penalty_for_critical_path_worsening = penalty_for_critical_path_worsening
        self.lb = initial_schedule.get_lower_bound_makespan()
        self.mb = initial_schedule.get_maximal_bound_makespan()

    def calculate_reward(self, reward_data: dict[str, Any]) -> float:
        clbs_before = list(
            reward_data["state_before"].get_operation_lower_bounds().values()
        )
        clbs_after = list(
            reward_data["state_after"].get_operation_lower_bounds().values()
        )

        if not clbs_before or not clbs_after:
            return 0.0

        # Compute spread (max - mean) before and after
        max_before = max(clbs_before)
        mean_before = sum(clbs_before) / len(clbs_before)
        spread_before = max_before - mean_before

        max_after = max(clbs_after)
        mean_after = sum(clbs_after) / len(clbs_after)
        spread_after = max_after - mean_after

        if self.normalization == "initial_lb":
            reward = (spread_before - spread_after) / self.lb
        elif self.normalization == "initial_max":
            reward = (spread_before - spread_after) / self.mb
        elif self.normalization == "current_max":
            # Normalize each spread by its current maximum CLB.
            # This converts the spread to a unitless ratio (0 to 1) representing balance.
            # spread / max = (max - mean) / max = 1 - (mean / max)
            reward = spread_before / max_before - spread_after / max_after
        else:
            # Original unnormalized behavior
            reward = spread_before - spread_after

        if self.penalty_for_critical_path_worsening:
            if spread_after < spread_before:
                if max_after > max_before:
                    # Balanced load but critical path worsened
                    reward = -0.5

        return reward * self.scaling_factor

    @property
    def name(self) -> str:
        return "operation_lower_bound_spread_normalized"


@REWARD_REGISTRY.register("q90_gap_normalized")
class Q90GapNormalizedReward(RewardFunction):
    """
    Reward based on normalized q90 gap: (max(LB) - q90(LB)) / max(LB).
    """

    def __init__(
        self,
        initial_schedule: Schedule,
        quantile: float = 90.0,
        scaling_factor: float = 1.0,
        **kwargs,
    ):
        """
        Args:
            initial_schedule: Initial schedule for the episode
            quantile: Percentile to use (default 90.0 for q90)
            scaling_factor: Multiplicative scaling for reward magnitude
        """
        super().__init__(initial_schedule, **kwargs)
        self.quantile = quantile
        self.scaling_factor = scaling_factor
        self.lb = initial_schedule.get_lower_bound_makespan()

    def calculate_reward(self, reward_data: dict[str, Any]) -> float:
        clbs_before = list(
            reward_data["state_before"].get_operation_lower_bounds().values()
        )
        clbs_after = list(
            reward_data["state_after"].get_operation_lower_bounds().values()
        )

        if not clbs_before or not clbs_after:
            return 0.0

        # Compute normalized gap for before state
        max_before = max(clbs_before)
        q_before = np.percentile(clbs_before, self.quantile)
        gap_before = (max_before - q_before) / max_before if max_before > 0 else 0.0

        # Compute normalized gap for after state
        max_after = max(clbs_after)
        q_after = np.percentile(clbs_after, self.quantile)
        gap_after = (max_after - q_after) / max_after if max_after > 0 else 0.0

        # Positive reward for gap reduction (increased concentration)
        reward = (gap_before - gap_after) * self.scaling_factor
        return reward

    @property
    def name(self) -> str:
        return "q90_gap_normalized"


@REWARD_REGISTRY.register("fraction_near_critical")
class FractionNearCriticalReward(RewardFunction):
    """
    Reward based on fraction of operations within threshold of max(LB).
    """

    def __init__(
        self,
        initial_schedule: Schedule,
        threshold: float = 0.95,
        scaling_factor: float = 1.0,
        **kwargs,
    ):
        """
        Args:
            initial_schedule: Initial schedule for the episode
            threshold: Fraction of max(LB) to define "near-critical" (default 0.95)
                      e.g., 0.95 means operations with LB >= 0.95 * max(LB)
            scaling_factor: Multiplicative scaling for reward magnitude
        """
        super().__init__(initial_schedule, **kwargs)
        self.threshold = threshold
        self.scaling_factor = scaling_factor
        self.lb = initial_schedule.get_lower_bound_makespan()

    def calculate_reward(self, reward_data: dict[str, Any]) -> float:
        clbs_before = list(
            reward_data["state_before"].get_operation_lower_bounds().values()
        )
        clbs_after = list(
            reward_data["state_after"].get_operation_lower_bounds().values()
        )

        if not clbs_before or not clbs_after:
            return 0.0

        # Compute fraction near-critical for before state
        max_before = max(clbs_before)
        near_critical_before = sum(
            1 for v in clbs_before if v >= self.threshold * max_before
        )
        fraction_before = near_critical_before / len(clbs_before)

        # Compute fraction near-critical for after state
        max_after = max(clbs_after)
        near_critical_after = sum(
            1 for v in clbs_after if v >= self.threshold * max_after
        )
        fraction_after = near_critical_after / len(clbs_after)

        # Positive reward for increased fraction (more focus on critical ops)
        reward = (fraction_after - fraction_before) * self.scaling_factor
        return reward

    @property
    def name(self) -> str:
        return "fraction_near_critical"


@REWARD_REGISTRY.register("reward_mixer")
class RewardMixer(RewardFunction):
    """
    Combines multiple reward functions by summing their weighted outputs.
    Allows mixing dense and sparse reward signals.
    """

    def __init__(
        self,
        initial_schedule: Schedule,
        reward_functions: list[str],
        weights: list[float] | None = None,
        reward_specific_kwargs: dict[str, dict[str, Any]] | None = None,
        **kwargs,
    ):
        super().__init__(initial_schedule, **kwargs)
        self.weights = weights if weights else [1.0] * len(reward_functions)
        self.reward_specific_kwargs = reward_specific_kwargs or {}

        if len(self.weights) != len(reward_functions):
            raise ValueError("Number of weights must match number of reward functions")

        self.rewards = []
        # Note: REWARD_REGISTRY is used to instantiate mixed rewards.
        for name in reward_functions:
            if name not in REWARD_REGISTRY:
                raise ValueError(f"Unknown reward function '{name}'")

            # Combine global kwargs with specific kwargs
            # Specific kwargs override global ones
            current_kwargs = kwargs.copy()
            if name in self.reward_specific_kwargs:
                current_kwargs.update(self.reward_specific_kwargs[name])

            reward_cls = REWARD_REGISTRY.get(name)
            self.rewards.append(reward_cls(initial_schedule, **current_kwargs))

    def calculate_reward(self, reward_data: dict[str, Any]) -> float:
        total_reward = 0.0
        for reward_func, weight in zip(self.rewards, self.weights):
            total_reward += weight * reward_func.calculate_reward(reward_data)
        return total_reward

    @property
    def name(self) -> str:
        return "reward_mixer"


@REWARD_REGISTRY.register("waiting_jobs")
class WaitingJobsReward(RewardFunction):
    """
    Dense reward based on the number of waiting jobs.
    r(g_t, g_{t+1}) = -(number of waiting jobs at time t)
    """

    def calculate_reward(self, reward_data: dict[str, Any]) -> float:
        schedule_before: Schedule = reward_data["state_before"]
        # Waiting jobs are those that are eligible to be scheduled
        num_waiting_jobs = len(schedule_before.get_eligible_jobs())
        return -float(num_waiting_jobs)

    @property
    def name(self) -> str:
        return "waiting_jobs"


# Registry of available reward functions (deprecated, use REWARD_REGISTRY)
REWARD_FUNCTIONS = REWARD_REGISTRY._registry


def get_reward_function(
    name: str, initial_schedule: Schedule, **kwargs
) -> RewardFunction:
    """
    Factory function to create reward functions
    """
    reward_cls = REWARD_REGISTRY.get(name)
    return reward_cls(initial_schedule=initial_schedule, **kwargs)


def list_reward_functions() -> dict[str, str]:
    """Return a dict of available reward functions and their descriptions"""
    return {name: name for name in REWARD_REGISTRY.list_available()}
