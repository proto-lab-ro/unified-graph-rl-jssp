import torch
import torch.nn as nn
from torch_geometric.data import Batch, Data
from torch_geometric.nn import global_mean_pool

from jssp_gnn.modules.GINEncoder import GINEncoder
from jssp_gnn.utils.tensor_ops import concat_node_graph_features


class SharedGraphFeatureExtractor(torch.nn.Module):
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

    def forward(self, observation):
        """
        Process graph observation from NewGraphEnv.

        Args:
            observation: TensorDict with observation["graph_data"] containing Data object(s)

        Returns:
            tuple: (node_embeddings, graph_embedding, batch_indices)
        """
        # Extract graph_data from the nested observation structure

        graph_data_list = []

        if isinstance(observation, Data):
            # Direct Data object
            graph_data_list.append(observation)
        elif isinstance(observation, list):
            # Already a list of Data objects
            graph_data_list = observation
        else:
            raise ValueError(f"Unexpected observation type: {type(observation)}")

        # Batch the graphs
        batched_pyg_data = Batch.from_data_list(graph_data_list)

        # Apply GNN to get node embeddings
        node_embeddings = self.gnn(
            batched_pyg_data.x, batched_pyg_data.edge_index
        )  # (total_nodes, hidden_dim)

        # Aggregate to get graph-level embedding
        graph_embedding = self.aggregate(
            node_embeddings, batched_pyg_data.batch
        )  # (batch_size, hidden_dim)

        # Reshape node embeddings back to [batch_size, num_nodes_per_graph, hidden_dim]
        #! Assuming all graphs have the same number of nodes
        batch_size = batched_pyg_data.num_graphs
        num_nodes_per_graph = batched_pyg_data.x.size(0) // batch_size
        node_embeddings = node_embeddings.view(
            batch_size, num_nodes_per_graph, self.hidden_dim
        )

        # Squeeze batch dimension if single graph
        if graph_embedding.shape[0] == 1:
            graph_embedding = graph_embedding.squeeze(0)
            node_embeddings = node_embeddings.squeeze(0)

        return node_embeddings, graph_embedding, batched_pyg_data.batch


class SB3LikeActor(torch.nn.Module):
    """
    Policy network that uses graph features to produce action logits.
    Uses node embeddings to compute logits for each job.
    """

    def __init__(self, shared_extractor, num_jobs, num_operations, forward_type="job"):
        super().__init__()
        self.shared_extractor = shared_extractor
        self.num_jobs = num_jobs
        self.num_operations = num_operations
        self.forward_type = forward_type

        input_dim = (
            shared_extractor.features_dim * 2
        )  # concatenated node + graph features

        # Policy head: 2 hidden layers with 64 dims each
        self.policy_head = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.Tanh(),  # ReLU
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

        # Initialize weights
        for m in self.policy_head:
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _forward_to_operation_logits(self, node_embeddings, graph_embedding):
        num_operations = self.num_operations
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

        # Handle both batched [batch_size, num_nodes, hidden_dim] and unbatched [num_nodes, hidden_dim]
        # is_batched = node_embeddings.dim() == 3

        logits = self.policy_head(x)  # (batch_size, num_nodes, 1) or (num_nodes, 1)
        logits = logits.squeeze(-1)  # Remove last dimension

        return logits

    def _forward_to_job_logits(self, node_embeddings):
        is_batched = node_embeddings.dim() == 3
        if is_batched:
            batch_size, num_nodes, _ = node_embeddings.shape
            # Get logits for each node: (batch_size, num_nodes, 1)
            node_logits = self.policy_head(node_embeddings)

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
            num_nodes = node_embeddings.shape[0]

            # Get logits for each node: (num_nodes, 1)
            node_logits = self.policy_head(node_embeddings)

            # Group by job and aggregate
            num_ops_per_job = num_nodes // self.num_jobs

            # Reshape to (num_jobs, num_ops_per_job, 1) and aggregate
            node_logits = node_logits.view(self.num_jobs, num_ops_per_job, 1)
            logits = node_logits.mean(dim=1)

            # Remove last dimension: (num_jobs,)
            logits = logits.squeeze(-1)

        return logits

    def forward(self, observation):
        """
        Args:
            observation: Dict with 'graph_data' containing PyG Data object

        Returns:
            logits: (batch_size, num_jobs) or (num_jobs,) action logits
        """
        node_embeddings, graph_embedding, _ = self.shared_extractor(observation)

        match self.forward_type:
            case "operation":
                return self._forward_to_operation_logits(
                    node_embeddings, graph_embedding
                )
            case "job":
                return self._forward_to_job_logits(node_embeddings)
            case _:
                raise ValueError(f"Unknown forward_type: {self.forward_type}")


class SB3LikeCritic(torch.nn.Module):
    """
    Value network that uses graph-level features to estimate state value.
    """

    def __init__(self, shared_extractor):
        super().__init__()
        self.shared_extractor = shared_extractor

        input_dim = shared_extractor.features_dim

        # Policy head: 2 hidden layers with 64 dims each
        self.value_head = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.Tanh(),  # ReLU
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

        # Initialize weights
        for m in self.value_head:
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, observation):
        """
        Args:
            observation: Dict with 'graph_data' containing PyG Data object

        Returns:
            state_value: (batch_size, 1) estimated state value
        """
        _, graph_embedding, _ = self.shared_extractor(observation)

        # Use graph-level embedding for value estimation
        state_value = self.value_head(graph_embedding)

        return state_value.unsqueeze(0) if state_value.dim() == 1 else state_value
