import torch.nn as nn
from torch_geometric.nn import GINConv, HeteroConv


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


class HeteroGINEncoder(nn.Module):
    """
    Heterogeneous Graph Isomorphism Network (GIN) Encoder.

    Parameters
    ----------
    metadata : tuple
        Tuple (node_types, edge_types) describing the heterogeneous graph structure.
    input_dims : dict[str, int]
        Dictionary mapping node types to their input feature dimensions.
    hidden_dim : int
        Number of hidden units per GIN layer and output embedding size.
    k_layers : int
        Number of GIN layers to use.
    """

    def __init__(self, metadata, input_dims, hidden_dim, k_layers):
        super().__init__()
        self.node_types, self.edge_types = metadata

        # Input projection layers to align dimensions
        self.lin_dict = nn.ModuleDict()
        for node_type, in_dim in input_dims.items():
            self.lin_dict[node_type] = nn.Linear(in_dim, hidden_dim)

        self.convs = nn.ModuleList()
        for _ in range(k_layers):
            conv_dict = {}
            for edge_type in self.edge_types:
                # edge_type is (src_type, rel_type, dst_type)
                # GINConv MLP: hidden_dim -> hidden_dim
                mlp = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                )
                conv_dict[edge_type] = GINConv(mlp, train_eps=False)

            self.convs.append(HeteroConv(conv_dict, aggr="sum"))

        # Initialize weights
        self.apply(init_weights)

    def forward(self, x_dict, edge_index_dict):
        """
        Forward pass.

        Parameters
        ----------
        x_dict : dict[str, torch.Tensor]
            Dictionary of node features {node_type: features}.
        edge_index_dict : dict[tuple, torch.LongTensor]
            Dictionary of edge indices {edge_type: edge_index}.

        Returns
        -------
        dict[str, torch.Tensor]
            Dictionary of node embeddings {node_type: embeddings}.
        """
        # 1. Project inputs to hidden_dim
        x_dict_curr = {}
        for node_type, x in x_dict.items():
            if node_type in self.lin_dict:
                x_dict_curr[node_type] = self.lin_dict[node_type](x)
            else:
                # Pass through if no projection defined (assuming already hidden_dim or not used)
                x_dict_curr[node_type] = x

        # 2. Apply HeteroConv layers
        for conv in self.convs:
            x_dict_curr = conv(x_dict_curr, edge_index_dict)

        return x_dict_curr
