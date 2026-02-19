from abc import ABC, abstractmethod
from typing import Any, TypedDict

import numpy as np
from gymnasium import spaces
from torchrl.envs.libs.gym import convert_box_spec, convert_dict_spec

from jssp_core.domain.domains import ObservationType
from jssp_core.schedule import Schedule


class ObservationData(TypedDict, total=False):
    """
    Standardized structure for observation data in JSSP environments.

    This TypedDict defines the possible keys that an observation can contain.
    Different ObservationProviders will populate different subsets of these keys
    based on their observation type (FLAT, DICT, GRAPH, etc.).

    Keys:
        # Standard JSSP (FLAT / DICT)
        job_next_op (np.ndarray): Index of next operation for each job.
        job_ready_time (np.ndarray): Earliest time each job is ready.
        machine_ready_time (np.ndarray): Earliest time each machine is available.
        operation_action_mask (np.ndarray): Mask of valid job actions.

        # Graph-based JSSP (GRAPH / GRAPH_MATRIX)
        node_feats (np.ndarray): Feature matrix for nodes (operations).
        edge_index (np.ndarray): Graph connectivity in COO format.
        edge_attr (np.ndarray): Edge features.

        # Heterogeneous Graph
        node_feats_op (np.ndarray): Features for operation nodes.
        node_feats_machine (np.ndarray): Features for machine nodes.
        edge_index_op_precedence_op (np.ndarray): Precedence edges between operations.
        edge_index_op_assignment_machine (np.ndarray): Assignment edges (op -> machine).
        edge_index_machine_assignment_op (np.ndarray): Assignment edges (machine -> op).
    """

    # For DICT / FLAT observations (Standard JSSP)
    job_next_op: np.ndarray
    job_ready_time: np.ndarray
    machine_ready_time: np.ndarray
    operation_action_mask: np.ndarray

    # For TestObservationProvider / specific variants
    is_scheduled: np.ndarray
    machine: np.ndarray
    duration: np.ndarray
    machine_ready_times: np.ndarray
    is_eligible: np.ndarray

    # For GRAPH / GRAPH_MATRIX observations
    node_feats: np.ndarray
    edge_index: np.ndarray
    edge_attr: np.ndarray

    # For HETERO GRAPH observations
    node_feats_op: np.ndarray
    node_feats_machine: np.ndarray
    edge_index_op_precedence_op: np.ndarray
    edge_index_op_assignment_machine: np.ndarray
    edge_index_machine_assignment_op: np.ndarray


class ObservationProvider(ABC):
    """
    Abstract base class for JSSP observation providers.

    Observation providers are responsible for transforming the internal state
    of a `Schedule` into a format suitable for Reinforcement Learning agents.
    Subclasses must define the observation space and the mapping from schedule
    state to observation data.
    """

    def __init__(self, schedule: Schedule):
        """
        Initialize the observation provider.

        Args:
            schedule: A Schedule instance used to determine dimensions
                     (num_jobs, num_machines, etc.) for the observation space.
        """

        self.schedule = schedule
        self.num_jobs = schedule.num_jobs
        self.num_machines = schedule.num_machines
        self.num_operations = schedule.num_operations
        self.num_machines_wo_loc = self.num_machines

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Return a unique identifier for this provider.

        Returns:
            str: The provider name.
        """
        pass

    @property
    @abstractmethod
    def observation_type(self) -> ObservationType:
        """
        Return the structural type of observations (FLAT, DICT, GRAPH, etc.).

        Returns:
            ObservationType: The category of the observation.
        """
        pass

    @abstractmethod
    def get_observation_space(self) -> spaces.Space:
        """
        Define the gymnasium observation space.

        Returns:
            gymnasium.spaces.Space: The configured observation space.
        """
        pass

    @abstractmethod
    def get_observation(self, schedule: Schedule) -> ObservationData:
        """
        Extract observation data from a given schedule state.

        Args:
            schedule: The current Schedule object to observe.

        Returns:
            ObservationData: Dictionary containing the observation features.
        """
        pass

    def get_observation_space_trl(self):
        """
        Convert the gymnasium observation space to a TorchRL-compatible spec.

        Returns:
            CompositeSpec or TensorSpec: The TorchRL observation specification.
        """
        if self.observation_type == ObservationType.GRAPH:
            return convert_dict_spec(self.get_observation_space())

        elif self.observation_type == ObservationType.FLAT:
            return convert_box_spec(self.get_observation_space())

        elif self.observation_type == ObservationType.DICT:
            return convert_dict_spec(self.get_observation_space())

        elif self.observation_type == ObservationType.GRAPH_MATRIX:
            return convert_dict_spec(self.get_observation_space())

        else:
            raise ValueError(f"Unknown observation type: {self.observation_type}. ")

    def reset(self, schedule: Schedule) -> None:
        """
        Re-initialize the provider for a new episode or instance.

        Args:
            schedule: The new Schedule instance.
        """
        self.schedule = schedule
        self.num_jobs = schedule.num_jobs
        self.num_machines = schedule.num_machines
        self.num_operations = schedule.num_operations
        self.num_machines_wo_loc = self.num_machines

    def get_obervation_in_keys(self):
        """
        Get the list of keys expected in the observation dictionary.

        Returns:
            list[str]: List of key names.
        """
        obs_space = self.get_observation_space()
        if self.observation_type == ObservationType.FLAT:
            return []
        elif self.observation_type == ObservationType.DICT:
            return list(obs_space.spaces.keys())

        elif self.observation_type == ObservationType.GRAPH:
            raise NotImplementedError(
                "get_obervation_in_keys not implemented for GRAPH observation type"
            )
        elif self.observation_type == ObservationType.GRAPH_MATRIX:
            raise NotImplementedError(
                "get_obervation_in_keys not implemented for GRAPH_MATRIX observation type"
            )
        else:
            raise ValueError(f"Unknown observation type: {self.observation_type}. ")

    def validate_observation(self, observation: dict[str, Any]) -> bool:
        """
        Check if an observation dictionary conforms to the defined space.

        Args:
            observation: The observation data to validate.

        Returns:
            bool: True if valid, False otherwise.
        """
        try:
            return self.get_observation_space().contains(observation)
        except Exception:
            return False

    def get_observation_flattened(self, schedule: Schedule) -> np.ndarray:
        """
        Get the observation as a single flattened NumPy array.

        Args:
            schedule: The current Schedule object.

        Returns:
            np.ndarray: Flattened observation vector.
        """
        obs = self.get_observation(schedule)

        flat_obs = self.flatten_tensordict(obs)
        return flat_obs

    def flatten_tensordict(self, feature_dict):
        """
        Flatten all arrays in a dictionary and concatenate them.

        Args:
            feature_dict: Dictionary of NumPy arrays or Tensors.

        Returns:
            np.ndarray: Concatenated flat vector.
        """
        flat_tensors = []
        for key, value in feature_dict.items():
            flat_tensors.append(value.flatten())
        return np.concatenate(flat_tensors, dtype=np.float32)
