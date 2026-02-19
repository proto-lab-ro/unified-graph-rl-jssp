from abc import ABC, abstractmethod
from typing import Any

from jssp_core.schedule import Schedule


class RewardFunction(ABC):
    """
    Abstract base class for JSSP reward functions.

    Reward functions calculate the feedback signal given to an RL agent
    after an action is taken. They typically compare the state of the
    schedule before and after the action.
    """

    def __init__(self, initial_schedule: Schedule, **kwargs):
        """
        Initialize the reward function.

        Args:
            initial_schedule: The initial Schedule object for the episode,
                             often used to calculate baseline metrics like
                             theoretical lower bounds or estimated makespans.
            **kwargs: Additional keyword arguments for specific reward strategies.
        """
        self.initial_schedule = initial_schedule

    @abstractmethod
    def calculate_reward(self, reward_data: dict[str, Any]) -> float:
        """
        Calculate the reward value based on transition data.

        Args:
            reward_data: A dictionary containing the transition context:
                - "state_before" (Schedule): State before the action.
                - "state_after" (Schedule): State after the action.
                - "schedule" (Schedule): Alias for state_after.
                - "is_complete" (bool): Whether the episode is finished.
                - "action_valid" (bool): Whether the last action was valid.
                - "initial_est_makespan" (float, optional): Initial estimate.
                - "step_count" (int, optional): Current step in the episode.

        Returns:
            float: The calculated reward value.
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Return a unique identifier for this reward function.

        Returns:
            str: The reward function name.
        """
        pass
