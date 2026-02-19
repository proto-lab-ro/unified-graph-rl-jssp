import torch
from torch_geometric.nn import global_mean_pool

from jssp_core.modules.nets import ActorPolicyHead, CriticValueHead
from jssp_gnn.modules.GINEncoder import GINEncoder
from jssp_gnn.modules.HeteroGINEncoder import HeteroGINEncoder
from jssp_gnn.utils.tensor_ops import concat_node_graph_features


class SharedGraphFeatureExtractorMatrix(torch.nn.Module):
    """
    Extracts graph features using a GINEncoder.
    Produces:
    - node_embeddings: embeddings for all nodes (for policy logits)
    - graph_embedding: pooled graph representation (for value function)
    """

    def __init__(
        self,
        input_dim,
        hidden_dim=64,
        k_layers=3,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.features_dim = hidden_dim
        self.gnn = GINEncoder(input_dim, hidden_dim, k_layers)
        self.aggregate = global_mean_pool

    def _batch_graphs_from_tensor(self, node_feats, edge_index):
        """
        Manually batch multiple graphs from tensor format by creating a disjoint union.

        Args:
            node_feats: Batched node features tensor [batch_size, num_nodes, feature_dim]
            edge_index: Edge indices tensor [batch_size, 2, num_edges]
                       or [2, num_edges] if same structure for all graphs

        Returns:
            tuple: (batched_node_feats, batched_edge_index, batch_indices)
                - batched_node_feats: [total_nodes, feature_dim]
                - batched_edge_index: [2, total_edges]
                - batch_indices: [total_nodes] indicating which graph each node belongs to
        """
        batch_size = node_feats.size(0)
        num_nodes_per_graph = node_feats.size(1)
        feature_dim = node_feats.size(2)

        # Flatten node features: [batch_size, num_nodes, features] -> [batch_size * num_nodes, features]
        batched_node_feats = node_feats.view(-1, feature_dim)

        # Handle edge_index batching
        if edge_index.dim() == 3:
            # Edge index is [batch_size, 2, num_edges]
            num_edges = edge_index.size(2)
            batched_edge_indices = []

            for graph_idx in range(batch_size):
                offset = graph_idx * num_nodes_per_graph
                shifted_edge_index = edge_index[graph_idx] + offset
                batched_edge_indices.append(shifted_edge_index)

            batched_edge_index = torch.cat(batched_edge_indices, dim=1)
        else:
            # Edge index is [2, num_edges] - same structure for all graphs
            # Replicate and shift for each graph
            num_edges = edge_index.size(1)
            batched_edge_indices = []

            for graph_idx in range(batch_size):
                offset = graph_idx * num_nodes_per_graph
                shifted_edge_index = edge_index + offset
                batched_edge_indices.append(shifted_edge_index)

            batched_edge_index = torch.cat(batched_edge_indices, dim=1)

        # Create batch indices: [0,0,...,0, 1,1,...,1, ..., batch_size-1,...]
        batch_indices = torch.repeat_interleave(
            torch.arange(batch_size, dtype=torch.long, device=node_feats.device),
            num_nodes_per_graph,
        )

        return batched_node_feats, batched_edge_index, batch_indices

    def forward(self, node_feats, edge_index):
        """
        Process graph observation from NewGraphEnv.

        Args:
            node_feats: Node features tensor
                - Batched: [..., num_nodes, feature_dim]
            edge_index: Edge indices tensor
                - Batched: [..., 2, num_edges]

        Returns:
            tuple: (node_embeddings, graph_embedding, batch_indices)
        """
        # Handle arbitrary batch dimensions
        # Expected input: [B1, B2, ..., N, F]
        # We want to treat [B1, B2, ...] as a single batch dimension B_flat

        original_batch_shape = node_feats.shape[:-2]  # Everything before N, F
        is_batched = len(original_batch_shape) > 0

        if is_batched:
            # Flatten batch dimensions
            num_nodes = node_feats.shape[-2]
            feature_dim = node_feats.shape[-1]

            # Flatten to [B_flat, N, F]
            node_feats_flat = node_feats.flatten(0, -3)

            # Edge index needs similar flattening if it has batch dimensions
            if edge_index.dim() > 2:
                # Assuming edge_index follows same batch structure: [..., 2, E]
                # Flatten to [B_flat, 2, E]
                edge_index_flat = edge_index.flatten(0, -3)
            else:
                # Edge index is shared [2, E], no flattening needed
                edge_index_flat = edge_index

            # Batch multiple graphs from tensor format
            (
                node_feats_batched,
                edge_index_batched,
                batch_indices,
            ) = self._batch_graphs_from_tensor(node_feats_flat, edge_index_flat)

            # Apply GNN to get node embeddings
            node_embeddings_flat = self.gnn(
                node_feats_batched, edge_index_batched
            )  # (total_nodes, hidden_dim)

            # Aggregate to get graph-level embedding using batch indices
            graph_embedding_flat = self.aggregate(
                node_embeddings_flat, batch=batch_indices
            )  # (B_flat, hidden_dim)

            # Reshape outputs to original structure
            # node_embeddings: [..., N, hidden_dim]
            # graph_embedding: [..., hidden_dim]

            # node_embeddings_flat is [B_flat * N, hidden_dim].
            # We first view it as [B_flat, N, hidden_dim]
            node_embeddings = node_embeddings_flat.view(
                *original_batch_shape, num_nodes, self.hidden_dim
            )
            graph_embedding = graph_embedding_flat.view(
                *original_batch_shape, self.hidden_dim
            )

        else:
            # Single graph case (no batch dims)
            batch_indices = torch.zeros(
                node_feats.size(0), dtype=torch.long, device=node_feats.device
            )
            node_embeddings = self.gnn(node_feats, edge_index)
            graph_embedding = self.aggregate(node_embeddings, batch=batch_indices)

            if graph_embedding.shape[0] == 1:
                graph_embedding = graph_embedding.squeeze(0)
                node_embeddings = node_embeddings.squeeze(0)

        return node_embeddings, graph_embedding, batch_indices


class HeteroSharedGraphFeatureExtractorMatrix(torch.nn.Module):
    """
    Extracts graph features using a HeteroGINEncoder.
    Produces:
    - node_embeddings: embeddings for all nodes (for policy logits)
    - graph_embedding: pooled graph representation (for value function)
    """

    def __init__(
        self,
        metadata,
        input_dims,
        hidden_dim=64,
        k_layers=3,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.features_dim = hidden_dim
        self.gnn = HeteroGINEncoder(metadata, input_dims, hidden_dim, k_layers)
        self.aggregate = global_mean_pool

    def _batch_graphs_from_tensor(self, node_feats_dict, edge_index_dict):
        """
        Manually batch multiple graphs from tensor format by creating a disjoint union.
        """
        # Assume all node types have same batch size
        first_node_type = list(node_feats_dict.keys())[0]
        batch_size = node_feats_dict[first_node_type].size(0)

        batched_node_feats_dict = {}
        batch_indices_dict = {}

        # 1. Batch node features
        for node_type, feats in node_feats_dict.items():
            # feats: [batch_size, num_nodes, feature_dim]
            num_nodes_per_graph = feats.size(1)
            feature_dim = feats.size(2)

            # Flatten: [batch_size * num_nodes, feature_dim]
            batched_node_feats_dict[node_type] = feats.view(-1, feature_dim)

            # Create batch indices
            batch_indices_dict[node_type] = torch.repeat_interleave(
                torch.arange(batch_size, dtype=torch.long, device=feats.device),
                num_nodes_per_graph,
            )

        # 2. Batch edge indices
        batched_edge_index_dict = {}
        for edge_type, edge_index in edge_index_dict.items():
            src_type, _, dst_type = edge_type

            num_src_nodes = node_feats_dict[src_type].size(1)
            num_dst_nodes = node_feats_dict[dst_type].size(1)

            if edge_index.dim() == 3:
                # [batch_size, 2, num_edges]
                batched_indices = []
                for i in range(batch_size):
                    src_offset = i * num_src_nodes
                    dst_offset = i * num_dst_nodes

                    # Create offset tensor
                    offset = torch.tensor(
                        [[src_offset], [dst_offset]], device=edge_index.device
                    )
                    batched_indices.append(edge_index[i] + offset)

                batched_edge_index_dict[edge_type] = torch.cat(batched_indices, dim=1)
            else:
                # [2, num_edges] - same structure
                batched_indices = []
                for i in range(batch_size):
                    src_offset = i * num_src_nodes
                    dst_offset = i * num_dst_nodes

                    offset = torch.tensor(
                        [[src_offset], [dst_offset]], device=edge_index.device
                    )
                    batched_indices.append(edge_index + offset)

                batched_edge_index_dict[edge_type] = torch.cat(batched_indices, dim=1)

        return batched_node_feats_dict, batched_edge_index_dict, batch_indices_dict

    def _parse_observation(self, observation):
        """
        Parses the flat observation dictionary from LbGnnHeteroObservationProvider
        into structured x_dict and edge_index_dict.
        """
        # Mapping from observation keys to node/edge types
        # This assumes the specific naming convention of LbGnnHeteroObservationProvider

        x_dict = {
            "operation": observation["node_feats_op"],
            "machine": observation["node_feats_machine"],
        }

        edge_index_dict = {
            ("operation", "precedence", "operation"): observation[
                "edge_index_op_precedence_op"
            ],
            ("operation", "assignment", "machine"): observation[
                "edge_index_op_assignment_machine"
            ],
            ("machine", "assignment", "operation"): observation[
                "edge_index_machine_assignment_op"
            ],
        }

        return x_dict, edge_index_dict

    def forward(self, observation):
        """
        Process heterogeneous graph observation.

        Args:
            observation: Flat dictionary from LbGnnHeteroObservationProvider
        """
        # Parse flat observation to structured dicts
        node_feats_dict, edge_index_dict = self._parse_observation(observation)

        # Handle arbitrary batch dimensions
        first_node_type = list(node_feats_dict.keys())[0]
        original_batch_shape = node_feats_dict[first_node_type].shape[:-2]
        is_batched = len(original_batch_shape) > 0

        if is_batched:
            # Flatten batch dimensions for all node types
            flat_node_feats_dict = {}
            for nt, feats in node_feats_dict.items():
                flat_node_feats_dict[nt] = feats.flatten(0, -3)

            # Flatten batch dimensions for all edge types if needed
            flat_edge_index_dict = {}
            for et, indices in edge_index_dict.items():
                if indices.dim() > 2:
                    flat_edge_index_dict[et] = indices.flatten(0, -3)
                else:
                    flat_edge_index_dict[et] = indices

            # Batch
            (
                batched_node_feats,
                batched_edge_index,
                batch_indices_dict,
            ) = self._batch_graphs_from_tensor(
                flat_node_feats_dict, flat_edge_index_dict
            )

            # Apply HeteroGNN
            node_embeddings_dict_flat = self.gnn(batched_node_feats, batched_edge_index)

            # Aggregate
            target_node_type = (
                "operation"
                if "operation" in node_embeddings_dict_flat
                else list(node_embeddings_dict_flat.keys())[0]
            )
            graph_embedding_flat = self.aggregate(
                node_embeddings_dict_flat[target_node_type],
                batch=batch_indices_dict[target_node_type],
            )

            # Reshape Outputs
            node_embeddings_dict = {}
            for nt, embeddings in node_embeddings_dict_flat.items():
                num_nodes = node_feats_dict[nt].shape[-2]
                node_embeddings_dict[nt] = embeddings.view(
                    *original_batch_shape, num_nodes, -1
                )

            graph_embedding = graph_embedding_flat.view(
                *original_batch_shape, self.hidden_dim
            )

        else:
            # Single graph
            batch_indices_dict = {
                nt: torch.zeros(feats.size(0), dtype=torch.long, device=feats.device)
                for nt, feats in node_feats_dict.items()
            }
            node_embeddings_dict = self.gnn(node_feats_dict, edge_index_dict)
            target_node_type = (
                "operation"
                if "operation" in node_embeddings_dict
                else list(node_embeddings_dict.keys())[0]
            )
            graph_embedding = self.aggregate(
                node_embeddings_dict[target_node_type],
                batch=batch_indices_dict[target_node_type],
            )

            if graph_embedding.shape[0] == 1:
                graph_embedding = graph_embedding.squeeze(0)

        return node_embeddings_dict, graph_embedding, batch_indices_dict


class SB3LikeActorMatrix(torch.nn.Module):
    """
    Policy network that uses graph features to produce action logits.
    Uses node embeddings to compute logits for each job.
    """

    def __init__(self, shared_extractor, forward_type="job"):
        super().__init__()
        self.shared_extractor = shared_extractor
        self.forward_type = forward_type

        input_dim = (
            shared_extractor.features_dim * 2
        )  # concatenated node + graph features

        # Policy head: 2 hidden layers with 64 dims each
        self.policy_head = ActorPolicyHead(input_dim)

    def _forward_to_operation_logits(self, node_embeddings, graph_embedding, mask):
        num_operations = mask.shape[-1]
        # needed for added machine nodes -> therefore just take first num_operations nodes
        if node_embeddings.dim() == 3:
            operation_embeddings = node_embeddings[:, :num_operations, :]
        elif node_embeddings.dim() == 2:
            operation_embeddings = node_embeddings[:num_operations, :]
        else:
            raise ValueError(
                f"Unexpected node_embeddings dimension: {node_embeddings.dim()}"
            )
        x = concat_node_graph_features(operation_embeddings, graph_embedding)
        # 36, 128
        # Handle both batched [batch_size, num_nodes, hidden_dim] and unbatched [num_nodes, hidden_dim]
        # is_batched = node_embeddings.dim() == 3

        logits = self.policy_head(x)  # (batch_size, num_nodes, 1) or (num_nodes, 1)
        # 36,1
        logits = logits.squeeze(-1)  # Remove last dimension

        return logits
        # 36

    def _forward_to_job_logits(self, node_embeddings, mask):
        # Determine number of operations
        num_operations = mask.shape[-1]

        # Slice to get only operation embeddings
        if node_embeddings.dim() == 3:
            operation_embeddings = node_embeddings[:, :num_operations, :]
        elif node_embeddings.dim() == 2:
            operation_embeddings = node_embeddings[:num_operations, :]
        else:
            raise ValueError(
                f"Unexpected node_embeddings dimension: {node_embeddings.dim()}"
            )

        is_batched = operation_embeddings.dim() == 3
        if is_batched:
            batch_size, num_nodes, _ = operation_embeddings.shape
            # Get logits for each node: (batch_size, num_nodes, 1)
            node_logits = self.policy_head(operation_embeddings)

            # Group by job and aggregate
            num_ops_per_job = num_nodes // self.num_jobs

            # Reshape to (batch_size, num_jobs, num_ops_per_job, 1)
            node_logits = node_logits.view(
                batch_size, self.num_jobs, num_ops_per_job, 1
            )

            # Aggregate over operations: (batch_size, num_jobs, 1)
            logits = node_logits.mean(dim=2)

            # Remove last dimension: (batch_size, num_jobs)
            logits = logits.squeeze(-1)
        else:
            # Unbatched case: (num_nodes, hidden_dim)
            num_nodes = operation_embeddings.shape[0]

            # Get logits for each node: (num_nodes, 1)
            node_logits = self.policy_head(operation_embeddings)

            # Group by job and aggregate
            num_ops_per_job = num_nodes // self.num_jobs

            # Reshape to (num_jobs, num_ops_per_job, 1) and aggregate
            node_logits = node_logits.view(self.num_jobs, num_ops_per_job, 1)
            logits = node_logits.mean(dim=1)

            # Remove last dimension: (num_jobs,)
            logits = logits.squeeze(-1)

        return logits

    def forward(self, node_feats, edge_index, mask=None):
        """
        Args:
            observation: Dict with 'graph_data' containing PyG Data object

        Returns:
            logits: (batch_size, num_jobs) or (num_jobs,) action logits
        """
        node_embeddings, graph_embedding, _ = self.shared_extractor(
            node_feats, edge_index
        )

        match self.forward_type:
            case "operation":
                return self._forward_to_operation_logits(
                    node_embeddings, graph_embedding, mask
                )
            case "job":
                return self._forward_to_job_logits(node_embeddings, mask)
            case _:
                raise ValueError(f"Unknown forward_type: {self.forward_type}")


class SB3LikeCriticMatrix(torch.nn.Module):
    """
    Value network that uses graph-level features to estimate state value.
    """

    def __init__(self, shared_extractor):
        super().__init__()
        self.shared_extractor = shared_extractor

        input_dim = shared_extractor.features_dim

        # Policy head: 2 hidden layers with 64 dims each
        self.value_head = CriticValueHead(input_dim)

    def forward(self, node_feats, edge_index):
        """
        Args:
            observation: Dict with 'graph_data' containing PyG Data object

        Returns:
            state_value: (batch_size, 1) estimated state value
        """
        _, graph_embedding, _ = self.shared_extractor(node_feats, edge_index)

        # Use graph-level embedding for value estimation
        state_value = self.value_head(graph_embedding)

        return state_value.unsqueeze(0) if state_value.dim() == 1 else state_value


class HeteroSB3LikeActorMatrix(torch.nn.Module):
    """
    Heterogeneous Policy network that uses graph features to produce action logits.
    Uses node embeddings to compute logits for each job.
    """

    def __init__(self, shared_extractor, forward_type="job"):
        super().__init__()
        self.shared_extractor = shared_extractor
        self.forward_type = forward_type
        self.num_jobs = None  # Will be set during forward if needed or passed in init

        input_dim = (
            shared_extractor.features_dim * 2
        )  # concatenated node + graph features

        # Policy head: 2 hidden layers with 64 dims each
        self.policy_head = ActorPolicyHead(input_dim)

    def _forward_to_operation_logits(self, node_embeddings_dict, graph_embedding, mask):
        # Get operation embeddings
        operation_embeddings = node_embeddings_dict["operation"]

        num_operations = mask.shape[-1]
        # Slice if needed (though usually operation nodes match num_operations)
        if operation_embeddings.dim() == 3:
            operation_embeddings = operation_embeddings[:, :num_operations, :]
        elif operation_embeddings.dim() == 2:
            operation_embeddings = operation_embeddings[:num_operations, :]

        x = concat_node_graph_features(operation_embeddings, graph_embedding)

        logits = self.policy_head(x)  # (batch_size, num_nodes, 1) or (num_nodes, 1)
        logits = logits.squeeze(-1)  # Remove last dimension

        return logits

    def _forward_to_job_logits(self, node_embeddings_dict, mask):
        # Get operation embeddings
        operation_embeddings = node_embeddings_dict["operation"]

        # Determine number of operations
        num_operations = mask.shape[-1]

        # Slice to get only operation embeddings
        if operation_embeddings.dim() == 3:
            operation_embeddings = operation_embeddings[:, :num_operations, :]
        elif operation_embeddings.dim() == 2:
            operation_embeddings = operation_embeddings[:num_operations, :]

        is_batched = operation_embeddings.dim() == 3

        # We need num_jobs to aggregate.
        # Assuming num_jobs is inferable or passed.
        # For now, let's assume we can infer it from mask shape if mask is (batch, num_jobs)
        # But mask here is usually action mask for jobs?
        # In JSSPEnv, action_mask is for jobs.

        if self.num_jobs is None:
            # Try to infer from mask if it corresponds to jobs
            if mask is not None:
                self.num_jobs = mask.shape[-1]
            else:
                raise ValueError("num_jobs must be known for job-level logits")

        if is_batched:
            batch_size, num_nodes, _ = operation_embeddings.shape
            # Get logits for each node: (batch_size, num_nodes, 1)
            node_logits = self.policy_head(operation_embeddings)

            # Group by job and aggregate
            num_ops_per_job = num_nodes // self.num_jobs

            # Reshape to (batch_size, num_jobs, num_ops_per_job, 1)
            node_logits = node_logits.view(
                batch_size, self.num_jobs, num_ops_per_job, 1
            )

            # Aggregate over operations: (batch_size, num_jobs, 1)
            logits = node_logits.mean(dim=2)

            # Remove last dimension: (batch_size, num_jobs)
            logits = logits.squeeze(-1)
        else:
            # Unbatched case: (num_nodes, hidden_dim)
            num_nodes = operation_embeddings.shape[0]

            # Get logits for each node: (num_nodes, 1)
            node_logits = self.policy_head(operation_embeddings)

            # Group by job and aggregate
            num_ops_per_job = num_nodes // self.num_jobs

            # Reshape to (num_jobs, num_ops_per_job, 1) and aggregate
            node_logits = node_logits.view(self.num_jobs, num_ops_per_job, 1)
            logits = node_logits.mean(dim=1)

            # Remove last dimension: (num_jobs,)
            logits = logits.squeeze(-1)

        return logits

    def forward(self, observation, mask=None):
        """
        Args:
            observation: Flat dictionary from LbGnnHeteroObservationProvider

        Returns:
            logits: (batch_size, num_jobs) or (num_jobs,) action logits
        """
        node_embeddings_dict, graph_embedding, _ = self.shared_extractor(observation)

        match self.forward_type:
            case "operation":
                return self._forward_to_operation_logits(
                    node_embeddings_dict, graph_embedding, mask
                )
            case "job":
                return self._forward_to_job_logits(node_embeddings_dict, mask)
            case _:
                raise ValueError(f"Unknown forward_type: {self.forward_type}")


class HeteroSB3LikeCriticMatrix(torch.nn.Module):
    """
    Heterogeneous Value network that uses graph-level features to estimate state value.
    """

    def __init__(self, shared_extractor):
        super().__init__()
        self.shared_extractor = shared_extractor

        input_dim = shared_extractor.features_dim

        # Policy head: 2 hidden layers with 64 dims each
        self.value_head = CriticValueHead(input_dim)

    def forward(self, observation):
        """
        Args:
            observation: Flat dictionary from LbGnnHeteroObservationProvider

        Returns:
            state_value: (batch_size, 1) estimated state value
        """
        _, graph_embedding, _ = self.shared_extractor(observation)

        # Use graph-level embedding for value estimation
        state_value = self.value_head(graph_embedding)

        return state_value.unsqueeze(0) if state_value.dim() == 1 else state_value
