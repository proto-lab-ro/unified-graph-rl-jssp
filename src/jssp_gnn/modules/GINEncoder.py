import torch.nn as nn
from torch_geometric.nn import GINConv


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


class GINEncoder(nn.Module):
    """
    Graph Isomorphism Network (GIN) Encoder for node representation learning.

    Parameters
    ----------
    input_dim : int
        Number of input features per node.
    hidden_dim : int
        Number of hidden units per GIN layer and output embedding size.
    k_layers : int
        Number of GIN layers to use.

    Attributes
    ----------
    convs : nn.ModuleList
        List of GINConv layers, each implemented as a small MLP (2-layer MLP with ReLU).
    """

    def __init__(self, input_dim, hidden_dim, k_layers):
        super().__init__()
        self.convs = nn.ModuleList(
            [
                GINConv(
                    nn.Sequential(
                        nn.Linear(input_dim if i == 0 else hidden_dim, hidden_dim),
                        nn.ReLU(),
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.ReLU(),
                    )
                )
                for i in range(k_layers)
            ]
        )
        for layer in self.convs:
            layer.apply(init_weights)

    def forward(self, x, edge_index):
        """
        Forward pass: applies K layers of GIN convolution to the input node features.

        Parameters
        ----------
        x : torch.Tensor
            Node features of shape (num_nodes, input_dim).
        edge_index : torch.LongTensor
            Graph connectivity in COO format, shape (2, num_edges).

        Returns
        -------
        torch.Tensor
            Node embeddings of shape (num_nodes, hidden_dim).
        """
        for conv in self.convs:
            x = conv(x, edge_index)
        return x  # node embeddings
