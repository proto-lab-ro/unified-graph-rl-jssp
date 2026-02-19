"""
Observation providers for JSSP environments.

This module provides a flexible system for creating different observation formats
without modifying the core environment logic. Different observation providers can
be easily switched to experiment with various state representations.
"""

import inspect
import logging
from typing import Any

import numpy as np
import torch
from gymnasium import spaces
from torchrl.data import (
    Bounded,
    Composite,
    NonTensor,
    UnboundedContinuous,
    UnboundedDiscrete,
)
from torchrl.envs.libs.gym import convert_box_spec, convert_dict_spec

from jssp_core.domain import GraphNormalization, ObservationType
from jssp_core.domain.observation import ObservationData, ObservationProvider
from jssp_core.registry import OBSERVATION_REGISTRY
from jssp_core.schedule import Schedule


logger = logging.getLogger(__name__)


@OBSERVATION_REGISTRY.register("dummy")
class DummyObservationProvider(ObservationProvider):
    """Dummy provider that returns an all-ones vector as the observation."""

    @property
    def name(self) -> str:
        return "dummy"

    @property
    def observation_type(self) -> ObservationType:
        return ObservationType.FLAT

    def get_observation_space(self) -> spaces.Space:
        """Space allowing values in [0,1], observation will be all ones."""
        return spaces.Box(0.0, 1.0, (2,), dtype=np.float32)

    def get_observation(self, schedule: Schedule) -> np.ndarray:
        """Return an array of ones matching the space."""
        return np.ones((2,), dtype=np.float32)

    def get_observation_flattened(self, schedule: Schedule) -> np.ndarray:
        """Return an array of ones matching the space."""
        return np.ones((2,), dtype=np.float32)

    def get_observation_space_trl(self):
        """TorchRL observation space for Dummy provider."""
        return Bounded(0.0, 1.0, shape=torch.Size([2]), dtype=torch.float32)

    def get_flattened_observation_space(self) -> spaces.Box:
        """Flattened version of dummy observation space."""
        return spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32)


@OBSERVATION_REGISTRY.register("default")
class DefaultObservationProvider(ObservationProvider):
    """
    Default observation provider that maintains compatibility with existing code.

    Provides the same observation format as the original JSSPEnv:
    - job_next_op: Next operation index for each job
    - job_ready_time: Ready time for each job
    - machine_ready_time: Ready time for each machine
    - operation_action_mask: Binary mask for eligible operations
    """

    def __init__(
        self,
        schedule: Schedule,
        observation_type: ObservationType = ObservationType.DICT,
    ):
        super().__init__(schedule)

        self._observation_type = observation_type
        if self._observation_type not in (ObservationType.DICT, ObservationType.FLAT):
            raise ValueError(
                f"Unsupported observation_type '{self._observation_type}'; only "
                f"{ObservationType.DICT} and {ObservationType.FLAT} are allowed."
            )

    @property
    def name(self) -> str:
        return "default"

    @property
    def observation_type(self) -> ObservationType:
        return self._observation_type

    def get_observation_space(self) -> spaces.Space:
        """Get the default observation space."""
        obs_dict_space = spaces.Dict(
            {
                "job_next_op": spaces.Box(0, np.inf, (self.num_jobs,), dtype=np.int32),
                "job_ready_time": spaces.Box(
                    0, np.inf, (self.num_jobs,), dtype=np.int16
                ),
                "machine_ready_time": spaces.Box(
                    0, np.inf, (self.num_machines_wo_loc,), dtype=np.int16
                ),
                "operation_action_mask": spaces.Box(
                    0, 1, (self.num_operations,), dtype=np.int8
                ),
            }
        )
        if self.observation_type == ObservationType.FLAT:
            return self.get_flattened_observation_space()

        elif self.observation_type == ObservationType.DICT:
            return obs_dict_space
        else:
            raise ValueError(
                f"Unsupported observation_type '{self.observation_type}'; only "
                f"{ObservationType.DICT} and {ObservationType.FLAT} are allowed."
            )

    def get_flattened_observation_space(self) -> spaces.Box:
        """Calculate total size for flattened observation vector."""
        total_size = self.num_jobs * 2 + self.num_machines_wo_loc + self.num_operations
        return spaces.Box(low=0.0, high=np.inf, shape=(total_size,), dtype=np.float32)

    def get_observation(self, schedule: Schedule) -> ObservationData | np.ndarray:
        """Generate the default observation format."""
        obs_dict: ObservationData = {
            "job_next_op": np.array(schedule.job_next_op, dtype=np.int32),
            "job_ready_time": np.array(schedule.job_ready_time, dtype=np.int16),
            "machine_ready_time": np.array(
                [
                    schedule.machine_ready_time.get(m, 0)
                    for m in range(self.num_machines_wo_loc)
                ],
                dtype=np.int16,
            ),
            "operation_action_mask": np.array(
                list(schedule.eligible_operations.values()), dtype=np.int8
            ),
        }

        if self.observation_type == ObservationType.FLAT:
            return self.flatten_tensordict(obs_dict)

        return obs_dict

    def get_observation_space_trl(self):
        """TorchRL compatible observation space."""
        if self.observation_type == ObservationType.FLAT:
            return convert_box_spec(self.get_observation_space())

        elif self.observation_type == ObservationType.DICT:
            return convert_dict_spec(self.get_observation_space())


@OBSERVATION_REGISTRY.register("test")
class TestObservationProvider(ObservationProvider):
    """
    Test observation provider for unit tests.
    """

    @property
    def name(self) -> str:
        return "test"

    @property
    def observation_type(self) -> ObservationType:
        return ObservationType.DICT

    def get_observation_space(self) -> spaces.Space:
        """Get the test observation space."""
        return spaces.Dict(
            {
                "is_scheduled": spaces.Box(0, 1, (self.num_operations,), dtype=np.int8),
                "machine": spaces.Box(0, np.inf, (self.num_operations,), dtype=np.int8),
                "duration": spaces.Box(
                    0, np.inf, (self.num_operations,), dtype=np.float32
                ),
                "job_ready_time": spaces.Box(
                    0, np.inf, (self.num_operations,), dtype=np.float32
                ),
                "machine_ready_times": spaces.Box(
                    0, np.inf, (self.num_operations,), dtype=np.float32
                ),
                "is_eligible": spaces.Box(0, 1, (self.num_operations,), dtype=np.int8),
            }
        )

    def get_observation(self, schedule: Schedule) -> ObservationData:
        """Generate the test observation format."""

        is_scheduled_ops = []
        duration_op = []
        is_eligible_op = []
        job_ready_times_op = []
        machine_op = []
        machine_ready_times_op = []

        scheduled_ops = schedule.scheduled
        job_next_ops = schedule.job_next_op
        eligible_jobs = schedule.get_eligible_jobs()
        job_ready_times = schedule.job_ready_time
        machine_ready_times = [
            schedule.machine_ready_time.get(m, 0)
            for m in range(self.num_machines_wo_loc)
        ]

        for job_idx, job in enumerate(schedule.instance):
            job_ready_time_val = job_ready_times[job_idx]
            next_op_idx = job_next_ops[job_idx]
            job_is_eligible = job_idx in eligible_jobs

            for op_idx, (machine, duration) in enumerate(job):
                # Check if operation is scheduled
                is_scheduled = (job_idx, op_idx) in scheduled_ops

                # Check if operation is eligible
                is_eligible = (op_idx == next_op_idx) and job_is_eligible

                # Build feature vector for this operation
                job_ready_times_op.append(
                    float(job_ready_time_val if is_scheduled else 0.0)
                )
                machine_ready_times_op.append(float(machine_ready_times[machine]))
                duration_op.append(float(duration))
                machine_op.append(machine)
                is_scheduled_ops.append(is_scheduled)
                is_eligible_op.append(is_eligible)

        res: ObservationData = {
            "is_scheduled": np.array(is_scheduled_ops, dtype=np.int8),
            "machine": np.array(machine_op, dtype=np.int8),
            "duration": np.array(duration_op, dtype=np.float32),
            "job_ready_time": np.array(job_ready_times_op, dtype=np.float32),
            "machine_ready_times": np.array(machine_ready_times_op, dtype=np.float32),
            "is_eligible": np.array(is_eligible_op, dtype=np.int8),
        }
        return res

    def get_observation_space_trl(self):
        """TorchRL composite spec for test provider."""
        obs_space_tl = Composite(
            {
                "is_scheduled": Bounded(0, 1, (self.num_operations,), dtype=torch.int8),
                "machine": UnboundedDiscrete((self.num_operations,), dtype=torch.int8),
                "duration": UnboundedContinuous(
                    (self.num_operations,), dtype=torch.float32
                ),
                "job_ready_time": UnboundedContinuous(
                    (self.num_operations,), dtype=torch.float32
                ),
                "machine_ready_times": UnboundedContinuous(
                    (self.num_operations,), dtype=torch.float32
                ),
                "is_eligible": Bounded(0, 1, (self.num_operations,), dtype=torch.int8),
            }
        )
        return obs_space_tl


@OBSERVATION_REGISTRY.register("minimal")
class MinimalObservationProvider(ObservationProvider):
    """
    Minimal observation provider with only essential information.

    Provides:
    - job_ready_time: Ready time for each job
    - operation_action_mask: Binary mask for eligible jobs (not operations)
    """

    def __init__(
        self,
        schedule: Schedule,
        observation_type: ObservationType = ObservationType.DICT,
    ):
        super().__init__(schedule)

        self._observation_type = observation_type
        if self._observation_type not in (ObservationType.DICT, ObservationType.FLAT):
            raise ValueError(
                f"Unsupported observation_type '{self._observation_type}'; only "
                f"{ObservationType.DICT} and {ObservationType.FLAT} are allowed."
            )

    @property
    def name(self) -> str:
        return "minimal"

    @property
    def observation_type(self) -> ObservationType:
        return self._observation_type

    def get_observation_space(self) -> spaces.Space:
        """Get the minimal observation space."""
        obs_dict_space = spaces.Dict(
            {
                "job_ready_time": spaces.Box(
                    0, np.inf, (self.num_jobs,), dtype=np.int16
                ),
                "operation_action_mask": spaces.Box(
                    0, 1, (self.num_jobs,), dtype=np.int8
                ),
            }
        )

        if self.observation_type == ObservationType.FLAT:
            return self.get_flattened_observation_space()

        elif self.observation_type == ObservationType.DICT:
            return obs_dict_space
        else:
            raise ValueError(
                f"Unsupported observation_type '{self.observation_type}'; only "
                f"{ObservationType.DICT} and {ObservationType.FLAT} are allowed."
            )

    def get_flattened_observation_space(self) -> spaces.Box:
        """Calculate size for flattened minimal observation."""
        total_size = self.num_jobs * 2
        return spaces.Box(low=0.0, high=np.inf, shape=(total_size,), dtype=np.float32)

    def get_observation(self, schedule: Schedule) -> ObservationData | np.ndarray:
        """Generate the minimal observation format."""
        # Create job-level action mask
        action_mask = np.zeros(self.num_jobs, dtype=np.int8)
        for job_id in range(self.num_jobs):
            if schedule.can_schedule_job(job_id):
                action_mask[job_id] = 1

        obs_dict: ObservationData = {
            "job_ready_time": np.array(schedule.job_ready_time, dtype=np.int16),
            "operation_action_mask": action_mask,
        }
        if self.observation_type == ObservationType.FLAT:
            return self.flatten_tensordict(obs_dict)

        return obs_dict

    def get_observation_space_trl(self):
        """TorchRL compatible observation space."""
        if self.observation_type == ObservationType.FLAT:
            return convert_box_spec(self.get_observation_space())

        elif self.observation_type == ObservationType.DICT:
            return convert_dict_spec(self.get_observation_space())


@OBSERVATION_REGISTRY.register("normalized")
class NormalizedObservationProvider(ObservationProvider):
    """
    Normalized observation provider that scales values to [0, 1] range.

    Provides the same information as default but with normalized values:
    - All time-based values are normalized by estimated completion time
    - Progress values are naturally in [0, 1] range
    """

    def __init__(
        self,
        schedule: Schedule,
        max_time_horizon: int | None = None,
        observation_type: ObservationType = ObservationType.DICT,
    ):
        """
        Initialize normalized observation provider.

        Args:
            schedule: Initial schedule instance
            max_time_horizon: Maximum time horizon for normalization (if None, uses estimated completion)
        """
        super().__init__(schedule)
        self.max_time_horizon = max_time_horizon

        self._observation_type = observation_type
        if self._observation_type not in (ObservationType.DICT, ObservationType.FLAT):
            raise ValueError(
                f"Unsupported observation_type '{self._observation_type}'; only "
                f"{ObservationType.DICT} and {ObservationType.FLAT} are allowed."
            )

    @property
    def name(self) -> str:
        return "normalized"

    @property
    def observation_type(self) -> ObservationType:
        return self._observation_type

    def get_observation_space(self) -> spaces.Space:
        """Get the normalized observation space (all values in [0, 1])."""
        obs_dict_space = spaces.Dict(
            {
                "job_next_op": spaces.Box(0, 1, (self.num_jobs,), dtype=np.float32),
                "job_ready_time": spaces.Box(0, 1, (self.num_jobs,), dtype=np.float32),
                "machine_ready_time": spaces.Box(
                    0, 1, (self.num_machines_wo_loc,), dtype=np.float32
                ),
                # "operation_action_mask": spaces.Box(
                #     0, 1, (self.num_operations,), dtype=np.int8
                # ),
            }
        )
        if self.observation_type == ObservationType.FLAT:
            return self.get_flattened_observation_space()

        elif self.observation_type == ObservationType.DICT:
            return obs_dict_space
        else:
            raise ValueError(
                f"Unsupported observation_type '{self.observation_type}'; only "
                f"{ObservationType.DICT} and {ObservationType.FLAT} are allowed."
            )

    def get_flattened_observation_space(self) -> spaces.Box:
        total_size = self.num_machines_wo_loc + self.num_jobs * 2
        return spaces.Box(low=0.0, high=1, shape=(total_size,), dtype=np.float32)

    def get_observation(self, schedule: Schedule) -> ObservationData:
        """Generate the normalized observation format."""
        # Determine normalization factor
        if self.max_time_horizon is not None:
            time_norm = self.max_time_horizon
        else:
            time_norm = max(
                schedule.estimate_completion_time(), 1
            )  # Avoid division by zero

        # Normalize job next operation as progress (0-1)
        job_next_op_progress = np.zeros(self.num_jobs, dtype=np.float32)
        for job_id in range(self.num_jobs):
            total_ops = len(schedule.instance[job_id])
            if total_ops > 0:
                job_next_op_progress[job_id] = schedule.job_next_op[job_id] / total_ops
            else:
                job_next_op_progress[job_id] = 1.0

        # Normalize time values
        job_ready_time_norm = (
            np.array(schedule.job_ready_time, dtype=np.float32) / time_norm
        )
        machine_ready_time_norm = (
            np.array(
                [
                    schedule.machine_ready_time.get(m, 0)
                    for m in range(self.num_machines_wo_loc)
                ],
                dtype=np.float32,
            )
            / time_norm
        )

        # Keep mask as-is (already binary)
        # operation_action_mask = np.array(
        #     list(schedule.eligible_operations.values()), dtype=np.int8
        # )

        obs_dict: ObservationData = {
            "job_next_op": job_next_op_progress,
            "job_ready_time": job_ready_time_norm,
            "machine_ready_time": machine_ready_time_norm,
            # "operation_action_mask": operation_action_mask,
        }
        if self.observation_type == ObservationType.FLAT:
            return self.flatten_tensordict(obs_dict)

        elif self.observation_type == ObservationType.DICT:
            return obs_dict

    def get_observation_space_trl(self):
        if self.observation_type == ObservationType.FLAT:
            return convert_box_spec(self.get_observation_space())

        elif self.observation_type == ObservationType.DICT:
            return convert_dict_spec(self.get_observation_space())


@OBSERVATION_REGISTRY.register("operation_lower_bound")
class OperationLowerBoundObservationProvider(ObservationProvider):
    """
    Operation lower bound observation provider.

    Provides:
    - operation_lower_bounds: Lower bound completion time for each operation
    - operation_scheduled: Binary indicator if each operation is already scheduled
    - operation_action_mask: Binary mask for eligible operations (for compatibility)
    """

    def __init__(
        self,
        schedule: Schedule,
        normalize: bool = True,
        observation_type: ObservationType = ObservationType.DICT,
    ):
        """
        Initialize operation lower bound observation provider.

        Args:
            schedule: Initial schedule instance
            normalize: Whether to normalize lower bounds by estimated completion time
        """
        super().__init__(schedule)
        self.normalize = normalize

        self._observation_type = observation_type
        if self._observation_type not in (ObservationType.DICT, ObservationType.FLAT):
            raise ValueError(
                f"Unsupported observation_type '{self._observation_type}'; only "
                f"{ObservationType.DICT} and {ObservationType.FLAT} are allowed."
            )

    @property
    def name(self) -> str:
        return "operation_lower_bound"

    @property
    def observation_type(self) -> ObservationType:
        return self._observation_type

    def get_flattened_observation_space(self) -> spaces.Box:
        total_size = self.num_operations * 3
        return spaces.Box(low=0.0, high=np.inf, shape=(total_size,), dtype=np.float32)

    def get_observation_space(self) -> spaces.Space:
        """Get the operation lower bound observation space."""
        if self.normalize:
            lower_bound_space = spaces.Box(
                0, 1, (self.num_operations,), dtype=np.float32
            )
        else:
            lower_bound_space = spaces.Box(
                0, np.inf, (self.num_operations,), dtype=np.float32
            )

        obs_dict_space = spaces.Dict(
            {
                "operation_lower_bounds": lower_bound_space,
                "operation_scheduled": spaces.Box(
                    0, 1, (self.num_operations,), dtype=np.int8
                ),
                "operation_action_mask": spaces.Box(
                    0, 1, (self.num_operations,), dtype=np.int8
                ),
            }
        )

        if self.observation_type == ObservationType.FLAT:
            return self.get_flattened_observation_space()

        elif self.observation_type == ObservationType.DICT:
            return obs_dict_space
        else:
            raise ValueError(
                f"Unsupported observation_type '{self.observation_type}'; only "
                f"{ObservationType.DICT} and {ObservationType.FLAT} are allowed."
            )

    def get_observation(self, schedule: Schedule) -> ObservationData:
        """Generate the operation lower bound observation format."""
        # Get lower bounds for all operations
        lower_bounds_dict = schedule.get_operation_lower_bounds()

        # Create arrays ordered by the same structure as eligible_operations
        operation_lower_bounds = []
        operation_scheduled = []

        # Iterate through operations in the same order as eligible_operations
        for job_id in range(self.num_jobs):
            for op_id in range(len(schedule.instance[job_id])):
                # Get lower bound for this operation
                lower_bound = lower_bounds_dict.get((job_id, op_id), 0.0)
                operation_lower_bounds.append(lower_bound)

                # Check if operation is scheduled
                is_scheduled = 1 if (job_id, op_id) in schedule.scheduled else 0
                operation_scheduled.append(is_scheduled)

        # Convert to numpy arrays
        operation_lower_bounds = np.array(operation_lower_bounds, dtype=np.float32)
        operation_scheduled = np.array(operation_scheduled, dtype=np.int8)

        # Normalize lower bounds if requested
        if self.normalize:
            # Use estimated completion time for normalization, avoid division by zero
            # time_norm = max(schedule.estimate_completion_time(), 1.0)
            time_norm = 55  #! Hardcoded for ft06
            operation_lower_bounds = operation_lower_bounds / time_norm

        # Keep the action mask for compatibility
        operation_action_mask = np.array(
            list(schedule.eligible_operations.values()), dtype=np.int8
        )

        obs_dict: ObservationData = {
            "operation_lower_bounds": operation_lower_bounds,
            "operation_scheduled": operation_scheduled,
            "operation_action_mask": operation_action_mask,
        }

        if self.observation_type == ObservationType.FLAT:
            return self.flatten_tensordict(obs_dict)

        elif self.observation_type == ObservationType.DICT:
            return obs_dict

    def get_observation_space_trl(self):
        if self.observation_type == ObservationType.FLAT:
            return convert_box_spec(self.get_observation_space())

        elif self.observation_type == ObservationType.DICT:
            return convert_dict_spec(self.get_observation_space())


@OBSERVATION_REGISTRY.register("gnn")
class GNNObservationProvider(ObservationProvider):
    """
    GNN observation provider that structures observations for Graph Neural Networks.

    Provides graph-structured observations with:
    - node_feats: Features for each operation (node) in the graph
    - edge_index: Edge connectivity for precedence and/or machine constraints

    The observations are padded to max_nodes and max_edges to maintain consistent shapes.
    """

    def __init__(
        self,
        schedule: Schedule,
        max_nodes: int | None = None,
        max_edges: int | None = None,
        include_machine_edges: bool = False,
        normalize_features: bool = True,
        both_directions: bool = False,
        self_loop: bool = False,
        observation_type: ObservationType = ObservationType.GRAPH,
    ):
        """
        Initialize GNN observation provider.

        Args:
            schedule: Initial schedule instance
            max_nodes: Maximum number of nodes for padding (if None, uses num_operations)
            max_edges: Maximum number of edges for padding (if None, computed from graph structure)
            include_machine_edges: Whether to include machine constraint edges in addition to precedence
            normalize_features: Whether to normalize time-based features
            both_directions: If True, adds reverse edges for precedence constraints
            self_loop: If True, adds self-loop edges for each node
        """
        super().__init__(schedule)
        self.max_nodes = max_nodes or self.num_operations
        self.include_machine_edges = include_machine_edges
        self.normalize_features = normalize_features
        self.both_directions = both_directions
        self.self_loop = self_loop
        self._observation_type = observation_type
        if self._observation_type not in (
            ObservationType.GRAPH,
            ObservationType.GRAPH_MATRIX,
        ):
            raise ValueError(
                f"Unsupported observation_type '{self._observation_type}'; only "
                f"{ObservationType.GRAPH} and {ObservationType.GRAPH_MATRIX} are allowed."
            )
        # Compute max_edges if not provided
        if max_edges is None:
            self.max_edges = self._estimate_max_edges()
        else:
            self.max_edges = max_edges

        # Create operation to node mapping
        self.op_node_id = {}
        node_id = 0
        for job_idx, job in enumerate(schedule.instance):
            for op_idx in range(len(job)):
                self.op_node_id[(job_idx, op_idx)] = node_id
                node_id += 1

    def _estimate_max_edges(self) -> int:
        """Estimate maximum number of edges needed for padding."""
        # Precedence edges: (ops_per_job - 1) per job
        precedence_edges = sum(max(0, len(job) - 1) for job in self.schedule.instance)

        if self.both_directions:
            precedence_edges *= 2

        # Machine edges: For each machine, create edges between all pairs of operations
        if self.include_machine_edges:
            machine_ops = {}
            for job_idx, job in enumerate(self.schedule.instance):
                for op_idx, (machine, _) in enumerate(job):
                    if machine not in machine_ops:
                        machine_ops[machine] = 0
                    machine_ops[machine] += 1

            # For n operations on same machine: n*(n-1) directed edges
            machine_edges = sum(n_ops * (n_ops - 1) for n_ops in machine_ops.values())
        else:
            machine_edges = 0

        # Self-loop edges
        self_loop_edges = self.num_operations if self.self_loop else 0

        total_edges = precedence_edges + machine_edges + self_loop_edges

        # Add some buffer (20%) for safety
        return int(total_edges * 1.2)

    @property
    def name(self) -> str:
        return "gnn"

    @property
    def observation_type(self) -> ObservationType:
        return self._observation_type

    def get_observation_space(self) -> spaces.Space:
        """Get the GNN observation space."""
        # Node features include:
        # - is_scheduled (1)
        # - job_id (1)
        # - op_id_in_job (1)
        # - machine_id (1)
        # - duration (1)
        # - job_ready_time (1)
        # - machine_ready_time (1)
        # Total: 7 features
        num_node_features = 7

        return spaces.Dict(
            {
                "node_feats": spaces.Box(
                    -np.inf,
                    np.inf,
                    (self.max_nodes, num_node_features),
                    dtype=np.float32,
                ),
                "edge_index": spaces.Box(
                    0, self.max_nodes - 1, (2, self.max_edges), dtype=np.int64
                ),
            }
        )

    def get_observation(self, schedule: Schedule) -> ObservationData:
        """Generate the GNN observation format."""
        # Build node features
        node_feats = self._build_node_features(schedule)

        # Build edge index
        edge_index = self._build_edge_index(schedule)

        obs: ObservationData = {
            "node_feats": node_feats,
            "edge_index": edge_index,
        }
        return obs

    def _build_node_features(self, schedule: Schedule) -> np.ndarray:
        """
        Build node feature matrix.

        Features per node:
        - is_scheduled: Whether the operation has been scheduled (0/1)
        - job_id: Job index (normalized if normalize_features=True)
        - op_id_in_job: Operation index within the job (normalized if normalize_features=True)
        - machine_id: Machine index (normalized if normalize_features=True)
        - duration: Processing time (normalized if normalize_features=True)
        - job_ready_time: Ready time for the job (normalized if normalize_features=True)
        - machine_ready_time: Ready time for the machine (normalized if normalize_features=True)
        """
        # Initialize feature matrix with zeros (for padding)
        node_feats = np.zeros((self.max_nodes, 7), dtype=np.float32)

        # Get schedule state
        scheduled_ops = schedule.scheduled
        job_ready_times = schedule.job_ready_time
        machine_ready_times = [
            schedule.machine_ready_time.get(m, 0)
            for m in range(self.num_machines_wo_loc)
        ]

        # Normalization factors
        if self.normalize_features:
            time_norm = max(schedule.estimate_completion_time(), 1.0)
            job_norm = max(self.num_jobs - 1, 1)
            machine_norm = max(self.num_machines_wo_loc - 1, 1)
        else:
            time_norm = 1.0
            job_norm = 1.0
            machine_norm = 1.0

        # Fill features for each operation
        node_idx = 0
        for job_idx, job in enumerate(schedule.instance):
            for op_idx, (machine, duration) in enumerate(job):
                # Feature 0: is_scheduled
                node_feats[node_idx, 0] = (
                    1.0 if (job_idx, op_idx) in scheduled_ops else 0.0
                )

                # Feature 1: job_id (normalized)
                node_feats[node_idx, 1] = job_idx / job_norm

                # Feature 2: op_id_in_job (normalized)
                max_ops_in_job = max(len(schedule.instance[job_idx]) - 1, 1)
                node_feats[node_idx, 2] = op_idx / max_ops_in_job

                # Feature 3: machine_id (normalized)
                node_feats[node_idx, 3] = machine / machine_norm

                # Feature 4: duration (normalized)
                node_feats[node_idx, 4] = duration / time_norm

                # Feature 5: job_ready_time (normalized)
                node_feats[node_idx, 5] = job_ready_times[job_idx] / time_norm

                # Feature 6: machine_ready_time (normalized)
                node_feats[node_idx, 6] = machine_ready_times[machine] / time_norm

                node_idx += 1

        return node_feats

    def _build_edge_index(self, schedule: Schedule) -> np.ndarray:
        """
        Build edge index matrix with precedence and optionally machine constraints.

        Returns:
            np.ndarray: Padded edge index array of shape (2, max_edges)
        """
        # Build precedence edges
        precedence_edges = schedule.build_precedence_edge_index(
            self_loop=self.self_loop, both_directions=self.both_directions
        )

        # Optionally add machine edges
        if self.include_machine_edges:
            machine_edges = schedule.build_machine_edge_index(self_loop=False)

            # Combine edges
            if precedence_edges.shape[1] > 0 and machine_edges.shape[1] > 0:
                edge_index = np.concatenate([precedence_edges, machine_edges], axis=1)
            elif machine_edges.shape[1] > 0:
                edge_index = machine_edges
            else:
                edge_index = precedence_edges
        else:
            edge_index = precedence_edges

        # Pad edge index to max_edges
        num_edges = edge_index.shape[1]
        if num_edges > self.max_edges:
            raise ValueError(
                f"Number of edges ({num_edges}) exceeds max_edges ({self.max_edges})."
            )
        elif num_edges < self.max_edges:
            # Pad with zeros
            padding = np.zeros((2, self.max_edges - num_edges), dtype=np.int64)
            edge_index = np.concatenate([edge_index, padding], axis=1)

        return edge_index

    def get_observation_space_trl(self):
        if self.observation_type == ObservationType.GRAPH:
            return NonTensor(
                shape=torch.Size(
                    [
                        1,
                    ]
                )
            )

        elif self.observation_type == ObservationType.GRAPH_MATRIX:
            return convert_dict_spec(self.get_observation_space())


@OBSERVATION_REGISTRY.register("lb_gnn")
class LbGnnObservationProvider(ObservationProvider):
    """
    GNN observation provider that structures observations for Graph Neural Networks.

    Assuming that after initialization, the schedule instance will not change size.

    Provides graph-structured observations with:
    - node_feats: Features for each operation (node) in the graph
    - edge_index: Edge connectivity for precedence and/or machine constraints

    The observations are padded to max_nodes and max_edges to maintain consistent shapes.
    """

    def __init__(
        self,
        schedule: Schedule,
        max_nodes: int | None = None,
        include_machine_edges: bool = False,
        time_norm: float = 1.0,
        both_directions: bool = False,
        self_loop: bool = False,
        pad_edges: bool = False,
        observation_type: ObservationType = ObservationType.GRAPH,
    ):
        """
        Initialize GNN observation provider.

        Assuming that after initialization, the schedule instance will not change size.

        Args:
            schedule: Initial schedule instance
            max_nodes: Maximum number of nodes for padding (if None, uses num_operations)
            include_machine_edges: Whether to include machine constraint edges in addition to precedence
            normalize_features: Whether to normalize time-based features
            both_directions: If True, adds reverse edges for precedence constraints
            self_loop: If True, adds self-loop edges for each node
        """
        super().__init__(schedule)
        self.max_nodes = max_nodes or self.num_operations
        self.include_machine_edges = include_machine_edges
        self.time_norm = time_norm
        self.both_directions = both_directions
        self.self_loop = self_loop
        self.max_edges = self._estimate_max_edges(schedule)
        self.pad_edges = pad_edges
        self._observation_type = observation_type
        if self._observation_type not in (
            ObservationType.GRAPH,
            ObservationType.GRAPH_MATRIX,
        ):
            raise ValueError(
                f"Unsupported observation_type '{self._observation_type}'; only "
                f"{ObservationType.GRAPH} and {ObservationType.GRAPH_MATRIX} are allowed."
            )
        # Create operation to node mapping
        self.op_node_id = {}
        node_id = 0
        for job_idx, job in enumerate(schedule.instance):
            for op_idx in range(len(job)):
                self.op_node_id[(job_idx, op_idx)] = node_id
                node_id += 1

    def _estimate_max_edges(self, schedule: Schedule) -> int:
        """
        Pre-calculate max_edges once since all instances share the same size.
        """
        precedence_edges = sum(max(len(job) - 1, 0) for job in schedule.instance)
        if self.both_directions:
            precedence_edges *= 2

        machine_edges = 0
        if self.include_machine_edges:
            machine_counts: dict[int, int] = {}
            for job in schedule.instance:
                for machine_id, _ in job:
                    machine_counts[machine_id] = machine_counts.get(machine_id, 0) + 1
            machine_edges = sum(n * (n - 1) for n in machine_counts.values())

        self_loop_edges = self.num_operations if self.self_loop else 0
        return precedence_edges + machine_edges + self_loop_edges

    @property
    def name(self) -> str:
        return "lb_gnn"

    @property
    def observation_type(self) -> ObservationType:
        return self._observation_type

    def get_observation_space(self) -> spaces.Space:
        """Get the GNN observation space."""
        num_node_features = 2
        return spaces.Dict(
            {
                "node_feats": spaces.Box(
                    -np.inf,
                    np.inf,
                    (self.max_nodes, num_node_features),
                    dtype=np.float32,
                ),
                "edge_index": spaces.Box(
                    0, self.max_nodes - 1, (2, self.max_edges), dtype=np.int64
                ),
            }
        )

    def get_observation(self, schedule: Schedule) -> dict[str, Any]:
        """Generate the GNN observation format."""
        # Build node features
        node_feats = self._build_node_features(schedule)

        # Build edge index
        edge_index = self._build_edge_index(schedule)

        return {
            "node_feats": node_feats,
            "edge_index": edge_index,
        }

    def _build_node_features(self, schedule: Schedule) -> np.ndarray:
        """
        Build node feature matrix.

        Features per node:
         - lb_per_operation: Lower bound completion time for the operation
         - is_scheduled: Whether the operation has been scheduled (0/1)

        """
        # Initialize feature matrix with zeros
        node_feats = np.zeros((self.num_operations, 2), dtype=np.float32)

        # Get schedule state
        scheduled_ops = schedule.scheduled
        lower_bounds_dict = schedule.get_operation_lower_bounds()
        # Fill features for each operation
        node_idx = 0
        for job_idx, job in enumerate(schedule.instance):
            for op_idx in range(len(job)):
                # Feature 0: lower bound completion time (normalized)
                lower_bound = lower_bounds_dict.get((job_idx, op_idx))
                if lower_bound is None:
                    raise ValueError(
                        f"Lower bound not found for operation (job {job_idx}, op {op_idx})"
                    )
                node_feats[node_idx, 0] = lower_bound / self.time_norm

                # Feature 1: is_scheduled
                node_feats[node_idx, 1] = (
                    1.0 if (job_idx, op_idx) in scheduled_ops else 0.0
                )

                node_idx += 1

        return node_feats

    def _build_edge_index(self, schedule: Schedule) -> np.ndarray:
        """
        Build edge index matrix with precedence and optionally machine constraints.

        Returns:
            np.ndarray: Padded edge index array of shape (2, max_edges)
        """
        # Build precedence edges
        precedence_edges = schedule.build_precedence_edge_index(
            self_loop=self.self_loop, both_directions=self.both_directions
        )

        # Optionally add machine edges
        if self.include_machine_edges:
            machine_edges = schedule.build_machine_edge_index(self_loop=False)

            # Combine edges
            if precedence_edges.shape[1] > 0 and machine_edges.shape[1] > 0:
                edge_index = np.concatenate([precedence_edges, machine_edges], axis=1)
            elif machine_edges.shape[1] > 0:
                edge_index = machine_edges
            else:
                edge_index = precedence_edges
        else:
            edge_index = precedence_edges

        # Pad edge index to max_edges
        num_edges = edge_index.shape[1]
        if num_edges > self.max_edges:
            raise ValueError(
                f"Number of edges {num_edges} exceeds max_edges {self.max_edges}"
            )

        return edge_index

    def get_observation_space_trl(self):
        if self.observation_type == ObservationType.GRAPH:
            return NonTensor(
                shape=torch.Size(
                    [
                        1,
                    ]
                )
            )

        elif self.observation_type == ObservationType.GRAPH_MATRIX:
            return convert_dict_spec(self.get_observation_space())


@OBSERVATION_REGISTRY.register("lb_maschinen_gnn")
class LbGnnMaschinenObservationProvider(ObservationProvider):
    """
    GNN observation provider that structures observations for Graph Neural Networks.

    Assuming that after initialization, the schedule instance will not change size.

    Provides graph-structured observations with:
    - node_feats: Features for each operation (node) in the graph
    - edge_index: Edge connectivity for precedence and/or machine constraints

    The observations are padded to max_nodes and max_edges to maintain consistent shapes.
    """

    def __init__(
        self,
        schedule: Schedule,
        max_nodes: int | None = None,
        include_machine_edges: bool = False,
        time_norm: float = 1.0,
        both_directions: bool = False,
        self_loop: bool = False,
        observation_type: ObservationType = ObservationType.GRAPH,
    ):
        """
        Initialize GNN observation provider.

        Args:
            schedule: Initial schedule instance
            max_nodes: Maximum number of nodes for padding (if None, uses num_operations)
            max_edges: Maximum number of edges for padding (if None, computed from graph structure)
            include_machine_edges: Whether to include machine constraint edges in addition to precedence
            normalize_features: Whether to normalize time-based features
            both_directions: If True, adds reverse edges for precedence constraints
            self_loop: If True, adds self-loop edges for each node
        """
        super().__init__(schedule)
        self.max_nodes = max_nodes or self.num_operations
        self.include_machine_edges = include_machine_edges
        self.time_norm = time_norm
        self.both_directions = both_directions
        self.self_loop = self_loop
        self.num_node_features = (
            2 + self.num_machines_wo_loc
        )  # lb, is_scheduled, one-hot machine
        self._observation_type = observation_type
        self.max_edges = self._estimate_max_edges(schedule)
        if self._observation_type not in (
            ObservationType.GRAPH,
            ObservationType.GRAPH_MATRIX,
        ):
            raise ValueError(
                f"Unsupported observation_type '{self._observation_type}'; only "
                f"{ObservationType.GRAPH} and {ObservationType.GRAPH_MATRIX} are allowed."
            )
        self.scale = self.schedule.get_lower_bound_makespan()

        # Create operation to node mapping
        self.op_node_id = {}
        node_id = 0
        for job_idx, job in enumerate(schedule.instance):
            for op_idx in range(len(job)):
                self.op_node_id[(job_idx, op_idx)] = node_id
                node_id += 1

    def _estimate_max_edges(self, schedule: Schedule) -> int:
        """
        Pre-calculate max_edges once since all instances share the same size.
        """
        precedence_edges = sum(max(len(job) - 1, 0) for job in schedule.instance)
        if self.both_directions:
            precedence_edges *= 2

        if self.self_loop:
            precedence_edges += self.num_operations

        return precedence_edges

    @property
    def name(self) -> str:
        return "lb_maschinen_gnn"

    @property
    def observation_type(self) -> ObservationType:
        return self._observation_type

    def get_observation_space(self) -> spaces.Space:
        """Get the GNN observation space."""
        return spaces.Dict(
            {
                "node_feats": spaces.Box(
                    -np.inf,
                    np.inf,
                    (self.max_nodes, self.num_node_features),
                    dtype=np.float32,
                ),
                "edge_index": spaces.Box(
                    0, self.max_nodes - 1, (2, self.max_edges), dtype=np.int64
                ),
            }
        )

    def get_observation(self, schedule: Schedule) -> ObservationData:
        """Generate the GNN observation format."""
        # Build node features
        node_feats = self._build_node_features(schedule)

        # Build edge index
        edge_index = self._build_edge_index(schedule)

        obs: ObservationData = {
            "node_feats": node_feats,
            "edge_index": edge_index,
        }
        return obs

    def _build_node_features(self, schedule: Schedule) -> np.ndarray:
        """
        Build the node feature matrix for the current scheduling state.

        Each node = one operation.

        Features per node:
            [0] lower_bound_norm : Lower-bound completion time (normalized)
            [1] is_scheduled     : 1 if operation is already scheduled, else 0
            [2:] machine_onehot  : One-hot encoding of machine assignment

        Notes:
        - Normalization uses a fixed per-episode scale (self.scale), e.g. initial max CLB.
        - The returned array has shape (num_operations, 2 + num_machines).
        """
        num_machines = self.num_machines_wo_loc
        num_ops = self.num_operations
        scheduled_ops = schedule.scheduled
        lower_bounds = schedule.get_operation_lower_bounds()

        # --- (1) Build CLB vector (lower-bound completion times) ------------------
        clb_vals = np.fromiter(
            (
                lower_bounds[(job_idx, op_idx)]
                for job_idx, job in enumerate(schedule.instance)
                for op_idx in range(len(job))
            ),
            dtype=np.float32,
            count=num_ops,
        )

        # Normalize by fixed per-episode scale
        clb_norm = clb_vals / self.scale

        # --- (2) Initialize feature matrix ----------------------------------------
        node_feats = np.zeros((num_ops, 2 + num_machines), dtype=np.float32)
        node_feats[:, 0] = clb_norm

        # --- (3) Fill is_scheduled + one-hot machine features ---------------------
        node_idx = 0
        for job_idx, job in enumerate(schedule.instance):
            for op_idx, (machine_id, _) in enumerate(job):
                node_feats[node_idx, 1] = float((job_idx, op_idx) in scheduled_ops)
                node_feats[node_idx, 2 + machine_id] = 1.0
                node_idx += 1

        return node_feats

    def _build_edge_index(self, schedule: Schedule) -> np.ndarray:
        """
        Build edge index matrix with precedence and optionally machine constraints.

        Returns:
            np.ndarray: Padded edge index array of shape (2, max_edges)
        """
        # Build precedence edges
        edge_index = schedule.build_precedence_edge_index(
            self_loop=self.self_loop, both_directions=self.both_directions
        )

        # Pad edge index to max_edges
        num_edges = edge_index.shape[1]
        if num_edges > self.max_edges:
            raise ValueError(
                f"Number of edges {num_edges} exceeds max_edges {self.max_edges}"
            )

        return edge_index

    def get_observation_space_trl(self):
        if self.observation_type == ObservationType.GRAPH:
            return NonTensor(
                shape=torch.Size(
                    [
                        1,
                    ]
                )
            )

        elif self.observation_type == ObservationType.GRAPH_MATRIX:
            return convert_dict_spec(self.get_observation_space())


@OBSERVATION_REGISTRY.register("lb_bipartite_gnn")
class LbGnnBipartiteObservationProvider(ObservationProvider):
    """
    GNN observation provider for the Operation–Machine Bipartite Graph.

    Assuming that after initialization, the schedule instance will not change size.

    Each node represents either:
      - an operation (job–machine pair), or
      - a machine (resource node).

    Edges connect:
      - successive operations in the same job (precedence edges)
      - each operation to its assigned machine (resource edges)

    Output:
      node_feats: [num_ops + num_machines, feat_dim]
      edge_index: [2, num_edges]

    Args:
        schedule: The schedule instance.
        max_nodes: Maximum number of nodes (for padding).
        include_precedence_edges: Whether to include precedence edges.
        both_directions: Whether to include edges in both directions.
        self_loop: Whether to include self-loops.
        observation_type: The type of observation to return.
        normalize: Normalization strategy. Can be a float for manual scaling (divides by this value),
                   or a GraphNormalization enum for automatic strategies (JOB, OPERATION).
                   Defaults to GraphNormalization.NONE (scale=1.0).
    """

    def __init__(
        self,
        schedule: Schedule,
        max_nodes: int | None = None,
        include_precedence_edges: bool = True,
        both_directions: bool = False,
        self_loop: bool = False,
        observation_type: ObservationType = ObservationType.GRAPH,
        normalize: float | GraphNormalization = GraphNormalization.NONE,
    ):
        super().__init__(schedule)

        # --- (1) Base dimensions ---------------------------------------------------
        # --- Varying per instance ---
        self.num_machines = schedule.num_machines
        self.num_ops = schedule.num_operations
        self.max_nodes = max_nodes or (self.num_ops + self.num_machines)

        # --- Constants
        self.include_precedence_edges = include_precedence_edges
        self.both_directions = both_directions
        self.self_loop = self_loop
        self._observation_type = observation_type

        # --- Varying per instance ---
        self.max_edges = self._estimate_max_edges(schedule)

        # Handle normalization configuration
        if isinstance(normalize, (int, float)):
            self.normalize = GraphNormalization.NONE
            self.scale = float(normalize)
        else:
            self.normalize = normalize
            self.scale = 1.0

        if self._observation_type not in (
            ObservationType.GRAPH,
            ObservationType.GRAPH_MATRIX,
        ):
            raise ValueError(
                f"Unsupported observation_type '{self._observation_type}'; only "
                f"{ObservationType.GRAPH} and {ObservationType.GRAPH_MATRIX} are allowed."
            )

        # --- (2) Shared feature schema --------------------------------------------
        # Operation features: [CLB_norm, is_scheduled]
        # Machine features:   [progress_norm, 0]
        # + 1 type flag (is_machine)
        # Total = 3 features per node (minimal version)
        self.num_node_features = 3

        # Mapping operation -> node index
        # --- Varying per instance ---
        self.op_node_id = {}
        node_id = 0
        for job_idx, job in enumerate(schedule.instance):
            for op_idx in range(len(job)):
                self.op_node_id[(job_idx, op_idx)] = node_id
                node_id += 1
        self.machine_offset = node_id  # machine nodes start after all operations

    def _estimate_max_edges(self, schedule: Schedule) -> int:
        """
        Pre-calculate max_edges once since all instances share the same size.
        """
        precedence_edges = 0
        if self.include_precedence_edges:
            precedence_edges = sum(max(len(job) - 1, 0) for job in schedule.instance)
            if self.both_directions:
                precedence_edges *= 2

        op_machine_edges = self.num_ops * 2  # op->machine and machine->op for each op
        self_loop_edges = self.num_ops + self.num_machines if self.self_loop else 0

        return precedence_edges + op_machine_edges + self_loop_edges

    @property
    def name(self) -> str:
        return "lb_bipartite_gnn"

    @property
    def observation_type(self) -> ObservationType:
        return self._observation_type

    def reset(self, schedule: Schedule) -> None:
        super().reset(schedule)
        self.num_machines = schedule.num_machines
        self.num_ops = schedule.num_operations
        self.max_nodes = self.num_ops + self.num_machines
        self.max_edges = self._estimate_max_edges(schedule)
        self.scale = self.schedule.get_lower_bound_makespan()

        # Create operation to node mapping
        self.op_node_id = {}
        node_id = 0
        for job_idx, job in enumerate(schedule.instance):
            for op_idx in range(len(job)):
                self.op_node_id[(job_idx, op_idx)] = node_id
                node_id += 1
        self.machine_offset = node_id  # machine nodes start after all operations

    def get_observation_space(self) -> spaces.Space:
        return spaces.Dict(
            {
                "node_feats": spaces.Box(
                    -np.inf,
                    np.inf,
                    (self.max_nodes, self.num_node_features),
                    dtype=np.float32,
                ),
                "edge_index": spaces.Box(
                    0, self.max_nodes - 1, (2, self.max_edges), dtype=np.int64
                ),
            }
        )

    def get_observation(self, schedule: Schedule) -> ObservationData:
        node_feats = self._build_node_features(schedule)
        edge_index = self._build_edge_index(schedule)
        obs: ObservationData = {"node_feats": node_feats, "edge_index": edge_index}
        return obs

    def _build_node_features(self, schedule: Schedule) -> np.ndarray:
        """
        Build features for all operation + machine nodes.

        Shared schema:
            [0] normalized_lower_bound (ops) / machine_progress (machines)
            [1] is_scheduled (ops) / 0 (machines)
            [2] is_machine flag (0=op, 1=machine)
        """
        num_nodes = self.num_ops + self.num_machines
        feats = np.zeros((num_nodes, self.num_node_features), dtype=np.float32)

        # ---- (A) Operation nodes -------------------------------------------------
        scheduled_ops = schedule.scheduled
        lower_bounds = schedule.get_operation_lower_bounds()
        if self.normalize == GraphNormalization.JOB:
            max_lb_per_job = self._get_max_lower_bound_per_job(lower_bounds, schedule)

        for job_idx, job in enumerate(schedule.instance):
            for op_idx, (machine_id, _) in enumerate(job):
                node = self.op_node_id[(job_idx, op_idx)]

                lb_norm = lower_bounds[(job_idx, op_idx)]
                if self.normalize == GraphNormalization.JOB:
                    lb_norm /= (
                        max_lb_per_job[job_idx] if max_lb_per_job[job_idx] > 0 else 1.0
                    )
                elif self.normalize == GraphNormalization.OPERATION:
                    lb_norm = lb_norm
                else:
                    lb_norm /= self.scale
                feats[node, 0] = lb_norm
                feats[node, 1] = float((job_idx, op_idx) in scheduled_ops)
                feats[node, 2] = 0.0  # is_machine flag
        if self.normalize == GraphNormalization.OPERATION:
            feats[:, 0] = self._normalize_by_all_operations(feats[:, 0])

        # ---- (B) Machine nodes ---------------------------------------------------
        for m_id in range(self.num_machines):
            node = self.machine_offset + m_id
            ops_for_machine = schedule.get_operations_on_machine(m_id)
            if len(ops_for_machine) > 0:
                num_sched = sum(1 for op in ops_for_machine if op in scheduled_ops)
                progress = num_sched / len(ops_for_machine)
            else:
                progress = 0.0

            feats[node, 0] = progress  # machine "load progress"
            feats[node, 1] = 0.0  # unused in constraint-based env
            feats[node, 2] = 1.0  # is_machine flag

        return feats

    def _normalize_by_all_operations(self, feature: np.ndarray) -> np.ndarray:
        max_value = np.max(feature) if np.max(feature) > 0 else 1.0
        min_value = np.min(feature) if np.min(feature) < 0 else -1.0
        normalized_feature = (feature - min_value) / (max_value - min_value)
        return normalized_feature

    def _get_max_lower_bound_per_job(
        self, lower_bounds: dict[tuple[int, int], float], schedule: Schedule
    ) -> dict[int, float]:
        max_lb_per_job = {}
        for job_idx, job in enumerate(schedule.instance):
            job_lbs = [
                lower_bounds.get((job_idx, op_idx), 0.0) for op_idx in range(len(job))
            ]
            max_lb_per_job[job_idx] = max(job_lbs) if job_lbs else 0.0
        return max_lb_per_job

    def _build_edge_index(self, schedule: Schedule) -> np.ndarray:
        """
        Build edge index for precedence and operation–machine edges.
        Returns (2, num_edges)
        """
        edges = []

        # ---- (A) Precedence edges ----------------------------------------------
        if self.include_precedence_edges:
            for job_idx, job in enumerate(schedule.instance):
                for op_idx in range(1, len(job)):
                    src = self.op_node_id[(job_idx, op_idx - 1)]
                    dst = self.op_node_id[(job_idx, op_idx)]
                    edges.append((src, dst))
                    if self.both_directions:
                        edges.append((dst, src))

        # ---- (B) Operation–Machine edges ----------------------------------------
        for job_idx, job in enumerate(schedule.instance):
            for op_idx, (machine_id, _) in enumerate(job):
                op_node = self.op_node_id[(job_idx, op_idx)]
                mach_node = self.machine_offset + machine_id
                edges.append((op_node, mach_node))
                edges.append((mach_node, op_node))

        # ---- (C) Self-loops -----------------------------------------------------
        if self.self_loop:
            for n in range(self.num_ops + self.num_machines):
                edges.append((n, n))

        edge_index = np.array(edges, dtype=np.int64).T  # [2, num_edges]

        if edge_index.shape[1] > self.max_edges:
            raise ValueError(
                f"Number of edges {edge_index.shape[1]} exceeds max_edges {self.max_edges}"
            )

        return edge_index

    def get_observation_space_trl(self):
        if self.observation_type == ObservationType.GRAPH:
            return NonTensor(
                shape=torch.Size(
                    [
                        1,
                    ]
                )
            )

        elif self.observation_type == ObservationType.GRAPH_MATRIX:
            return convert_dict_spec(self.get_observation_space())


@OBSERVATION_REGISTRY.register("lb_bipartite_non_overlapping_gnn")
class LbGnnBipartiteNonOverlappingObservationProvider(
    LbGnnBipartiteObservationProvider
):
    """
    GNN observation provider for the Operation–Machine Bipartite Graph with non-overlapping features.

    Features are separated into distinct channels for Operations and Machines to avoid semantic overlap.

    Node Features (Size 5):
    - Operation Nodes: [LB, Status, 0, 1, 0]
      - LB: Lower bound completion time (normalized)
      - Status: is_scheduled (0/1)
      - Type Flag: 1 for Operation (index 3)

    - Machine Nodes: [0, 0, Utilization, 0, 1]
      - Utilization: Machine progress/utilization
      - Type Flag: 1 for Machine (index 4)
    """

    @property
    def name(self) -> str:
        return "lb_bipartite_non_overlapping_gnn"

    def __init__(
        self,
        schedule: Schedule,
        max_nodes: int | None = None,
        include_precedence_edges: bool = True,
        both_directions: bool = False,
        self_loop: bool = False,
        observation_type: ObservationType = ObservationType.GRAPH,
        normalize: float | GraphNormalization = GraphNormalization.NONE,
    ):
        super().__init__(
            schedule,
            max_nodes=max_nodes,
            include_precedence_edges=include_precedence_edges,
            both_directions=both_directions,
            self_loop=self_loop,
            observation_type=observation_type,
            normalize=normalize,
        )
        self.num_node_features = 5

    def _build_node_features(self, schedule: Schedule) -> np.ndarray:
        """
        Build features for all operation + machine nodes with non-overlapping schema.

        Schema (Size 5):
        [0] LB (Ops) / 0 (Machines)
        [1] Status (Ops) / 0 (Machines)
        [2] 0 (Ops) / Utilization (Machines)
        [3] 1 (Ops) / 0 (Machines) - Op Type Flag
        [4] 0 (Ops) / 1 (Machines) - Machine Type Flag
        """
        num_nodes = self.num_ops + self.num_machines
        feats = np.zeros((num_nodes, self.num_node_features), dtype=np.float32)

        # ---- (A) Operation nodes -------------------------------------------------
        scheduled_ops = schedule.scheduled
        lower_bounds = schedule.get_operation_lower_bounds()

        if self.normalize == GraphNormalization.JOB:
            max_lb_per_job = self._get_max_lower_bound_per_job(lower_bounds, schedule)

        for job_idx, job in enumerate(schedule.instance):
            for op_idx, (machine_id, _) in enumerate(job):
                node = self.op_node_id[(job_idx, op_idx)]

                lb_norm = lower_bounds[(job_idx, op_idx)]
                if self.normalize == GraphNormalization.JOB:
                    lb_norm /= (
                        max_lb_per_job[job_idx] if max_lb_per_job[job_idx] > 0 else 1.0
                    )
                elif self.normalize == GraphNormalization.OPERATION:
                    pass  # Normalized later
                else:
                    lb_norm /= self.scale

                # [LB, Status, 0, 1, 0]
                feats[node, 0] = lb_norm
                feats[node, 1] = float((job_idx, op_idx) in scheduled_ops)
                feats[node, 3] = 1.0  # Op Type Flag

        if self.normalize == GraphNormalization.OPERATION:
            feats[:, 0] = self._normalize_by_all_operations(feats[:, 0])

        # ---- (B) Machine nodes ---------------------------------------------------
        for m_id in range(self.num_machines):
            node = self.machine_offset + m_id
            ops_for_machine = schedule.get_operations_on_machine(m_id)
            if len(ops_for_machine) > 0:
                num_sched = sum(1 for op in ops_for_machine if op in scheduled_ops)
                progress = num_sched / len(ops_for_machine)
            else:
                progress = 0.0

            # [0, 0, Utilization, 0, 1]
            feats[node, 2] = progress  # Machine Utilization
            feats[node, 4] = 1.0  # Machine Type Flag

        return feats


@OBSERVATION_REGISTRY.register("lb_hetero_gnn")
class LbGnnHeteroObservationProvider(ObservationProvider):
    """
    Heterogeneous GNN observation provider.

    Separates the graph into 'operation' and 'machine' nodes.

    Node Types:
      - "operation": Features [normalized_lower_bound, is_scheduled]
      - "machine":   Features [machine_progress]

    Edge Types:
      - ("operation", "precedence", "operation"): Precedence constraints
      - ("operation", "assignment", "machine"):   Operation assigned to machine
      - ("machine", "assignment", "operation"):   Machine assigned to operation (reverse)
    """

    def __init__(
        self,
        schedule: Schedule,
        include_precedence_edges: bool = True,
        both_directions: bool = False,
        observation_type: ObservationType = ObservationType.GRAPH,
        normalize: float | GraphNormalization = GraphNormalization.NONE,
    ):
        super().__init__(schedule)

        self.num_machines = schedule.num_machines
        self.num_ops = schedule.num_operations

        self.include_precedence_edges = include_precedence_edges
        self.both_directions = both_directions
        self._observation_type = observation_type

        # Handle normalization configuration
        if isinstance(normalize, (int, float)):
            self.normalize = GraphNormalization.NONE
            self.scale = float(normalize)
        else:
            self.normalize = normalize
            self.scale = 1.0

        if self._observation_type not in (
            ObservationType.GRAPH,
            ObservationType.GRAPH_MATRIX,
        ):
            raise ValueError(
                f"Unsupported observation_type '{self._observation_type}'; only "
                f"{ObservationType.GRAPH} and {ObservationType.GRAPH_MATRIX} are allowed."
            )

        # Mapping operation -> node index (0 to num_ops-1)
        self.op_node_id = {}
        node_id = 0
        for job_idx, job in enumerate(schedule.instance):
            for op_idx in range(len(job)):
                self.op_node_id[(job_idx, op_idx)] = node_id
                node_id += 1

        # Pre-calculate max edges
        self.max_precedence_edges = self._estimate_precedence_edges(schedule)
        self.max_assignment_edges = self.num_ops

    def _estimate_precedence_edges(self, schedule: Schedule) -> int:
        count = 0
        if self.include_precedence_edges:
            count = sum(max(len(job) - 1, 0) for job in schedule.instance)
            if self.both_directions:
                count *= 2
        return count

    @property
    def name(self) -> str:
        return "lb_hetero_gnn"

    @property
    def observation_type(self) -> ObservationType:
        return self._observation_type

    def reset(self, schedule: Schedule) -> None:
        super().reset(schedule)
        self.num_machines = schedule.num_machines
        self.num_ops = schedule.num_operations
        self.scale = self.schedule.get_lower_bound_makespan()

        # Re-map op nodes
        self.op_node_id = {}
        node_id = 0
        for job_idx, job in enumerate(schedule.instance):
            for op_idx in range(len(job)):
                self.op_node_id[(job_idx, op_idx)] = node_id
                node_id += 1

        self.max_precedence_edges = self._estimate_precedence_edges(schedule)
        self.max_assignment_edges = self.num_ops

    def get_observation_space(self) -> spaces.Space:
        return spaces.Dict(
            {
                "node_feats_op": spaces.Box(
                    -np.inf, np.inf, (self.num_ops, 2), dtype=np.float32
                ),
                "node_feats_machine": spaces.Box(
                    -np.inf, np.inf, (self.num_machines, 1), dtype=np.float32
                ),
                "edge_index_op_precedence_op": spaces.Box(
                    0,
                    self.num_ops - 1,
                    (2, self.max_precedence_edges),
                    dtype=np.int64,
                ),
                "edge_index_op_assignment_machine": spaces.Box(
                    0,
                    max(self.num_ops, self.num_machines),
                    (2, self.max_assignment_edges),
                    dtype=np.int64,
                ),
                "edge_index_machine_assignment_op": spaces.Box(
                    0,
                    max(self.num_ops, self.num_machines),
                    (2, self.max_assignment_edges),
                    dtype=np.int64,
                ),
            }
        )

    def get_observation(self, schedule: Schedule) -> ObservationData:
        node_feats_op, node_feats_machine = self._build_node_features(schedule)
        edge_indices = self._build_edge_indices(schedule)

        obs: ObservationData = {
            "node_feats_op": node_feats_op,
            "node_feats_machine": node_feats_machine,
            **edge_indices,
        }
        return obs

    def _build_node_features(self, schedule: Schedule) -> tuple[np.ndarray, np.ndarray]:
        # Operation features: [normalized_lower_bound, is_scheduled]
        feats_op = np.zeros((self.num_ops, 2), dtype=np.float32)

        scheduled_ops = schedule.scheduled
        lower_bounds = schedule.get_operation_lower_bounds()

        if self.normalize == GraphNormalization.JOB:
            max_lb_per_job = self._get_max_lower_bound_per_job(lower_bounds, schedule)

        for job_idx, job in enumerate(schedule.instance):
            for op_idx, (machine_id, _) in enumerate(job):
                node = self.op_node_id[(job_idx, op_idx)]

                lb_norm = lower_bounds[(job_idx, op_idx)]
                if self.normalize == GraphNormalization.JOB:
                    lb_norm /= (
                        max_lb_per_job[job_idx] if max_lb_per_job[job_idx] > 0 else 1.0
                    )
                elif self.normalize == GraphNormalization.OPERATION:
                    pass  # Normalized later -> see 5 lines below
                else:
                    lb_norm /= self.scale

                feats_op[node, 0] = lb_norm
                feats_op[node, 1] = float((job_idx, op_idx) in scheduled_ops)

        if self.normalize == GraphNormalization.OPERATION:
            feats_op[:, 0] = self._normalize_by_all_operations(feats_op[:, 0])

        # Machine features: [progress]
        feats_machine = np.zeros((self.num_machines, 1), dtype=np.float32)
        for m_id in range(self.num_machines):
            ops_for_machine = schedule.get_operations_on_machine(m_id)
            if len(ops_for_machine) > 0:
                num_sched = sum(1 for op in ops_for_machine if op in scheduled_ops)
                progress = num_sched / len(ops_for_machine)
            else:
                progress = 0.0
            feats_machine[m_id, 0] = progress

        return feats_op, feats_machine

    def _build_edge_indices(self, schedule: Schedule) -> dict[str, np.ndarray]:
        # Precedence edges
        edges_prec = []
        if self.include_precedence_edges:
            for job_idx, job in enumerate(schedule.instance):
                for op_idx in range(1, len(job)):
                    src = self.op_node_id[(job_idx, op_idx - 1)]
                    dst = self.op_node_id[(job_idx, op_idx)]
                    edges_prec.append((src, dst))
                    if self.both_directions:
                        edges_prec.append((dst, src))

        edge_index_prec = (
            np.array(edges_prec, dtype=np.int64).T
            if edges_prec
            else np.zeros((2, 0), dtype=np.int64)
        )

        # Assignment edges
        edges_assign_op_mach = []
        edges_assign_mach_op = []

        for job_idx, job in enumerate(schedule.instance):
            for op_idx, (machine_id, _) in enumerate(job):
                op_node = self.op_node_id[(job_idx, op_idx)]
                mach_node = machine_id  # Machine nodes are 0-indexed in their own space

                edges_assign_op_mach.append((op_node, mach_node))
                edges_assign_mach_op.append((mach_node, op_node))

        edge_index_op_mach = np.array(edges_assign_op_mach, dtype=np.int64).T
        edge_index_mach_op = np.array(edges_assign_mach_op, dtype=np.int64).T

        return {
            "edge_index_op_precedence_op": edge_index_prec,
            "edge_index_op_assignment_machine": edge_index_op_mach,
            "edge_index_machine_assignment_op": edge_index_mach_op,
        }

    def _normalize_by_all_operations(self, feature: np.ndarray) -> np.ndarray:
        max_value = np.max(feature) if np.max(feature) > 0 else 1.0
        min_value = np.min(feature) if np.min(feature) < 0 else -1.0
        normalized_feature = (feature - min_value) / (max_value - min_value)
        return normalized_feature

    def _get_max_lower_bound_per_job(
        self, lower_bounds: dict[tuple[int, int], float], schedule: Schedule
    ) -> dict[int, float]:
        max_lb_per_job = {}
        for job_idx, job in enumerate(schedule.instance):
            job_lbs = [
                lower_bounds.get((job_idx, op_idx), 0.0) for op_idx in range(len(job))
            ]
            max_lb_per_job[job_idx] = max(job_lbs) if job_lbs else 0.0
        return max_lb_per_job

    def get_observation_space_trl(self):
        if self.observation_type == ObservationType.GRAPH:
            return NonTensor(
                shape=torch.Size(
                    [
                        1,
                    ]
                )
            )

        elif self.observation_type == ObservationType.GRAPH_MATRIX:
            return convert_dict_spec(self.get_observation_space())


# Registry of available observation providers (deprecated, use OBSERVATION_REGISTRY)
OBSERVATION_PROVIDERS = OBSERVATION_REGISTRY._registry


def get_observation_provider(
    name: str, schedule: Schedule, **kwargs
) -> ObservationProvider:
    """
    Factory function to create observation providers.

    Args:
        name: Name of the observation provider.
        schedule: Schedule instance for initialization.
        **kwargs: Additional arguments for specific providers.

    Returns:
        ObservationProvider instance.

    Raises:
        ValueError: If provider name is not recognized.
    """
    provider_class = OBSERVATION_REGISTRY.get(name)

    # Inspect __init__ to see what it actually accepts
    sig = inspect.signature(provider_class.__init__)
    params = sig.parameters

    # Does the __init__ accept **kwargs?
    has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())

    if has_var_kw:
        # For providers that explicitly say "I accept **kwargs",
        # pass everything and let the class decide what to do.
        # (They can use the pop+warn pattern inside.)
        return provider_class(schedule, **kwargs)

    # Otherwise, limit kwargs to the declared parameters and warn on the rest
    accepted_names = {
        name
        for name, p in params.items()
        if name not in ("self", "schedule")
        and p.kind
        in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        )
    }

    filtered_kwargs = {k: v for k, v in kwargs.items() if k in accepted_names}
    unused_kwargs = {k: v for k, v in kwargs.items() if k not in accepted_names}

    if unused_kwargs:
        logger.warning(
            "Unused keyword arguments for provider '%s' in factory: %s",
            name,
            ", ".join(sorted(unused_kwargs.keys())),
        )

    return provider_class(schedule, **filtered_kwargs)
