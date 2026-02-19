import torch


def sum_param_weights(model):
    s = 0
    for param in model.parameters():
        s += param.detach().flatten().sum().item()
    return s


def concat_node_graph_features(X, graph_emb):
    """
    X: [N, d] or [B, N, d]
    graph_emb: [d] or [B, d]
    returns: [N, 2d] or [B, N, 2d]
    """

    # Case: unbatched graph_emb [d]
    if graph_emb.dim() == 1:
        if X.dim() == 2:  # X is [N, d]
            graph_emb_expanded = graph_emb.unsqueeze(0).expand(X.size(0), -1)  # [N, d]
        elif X.dim() == 3:  # X is [B, N, d]
            graph_emb_expanded = (
                graph_emb.unsqueeze(0).unsqueeze(1).expand(X.size(0), X.size(1), -1)
            )  # [B, N, d]

    # Case: batched graph_emb [B, d]
    elif graph_emb.dim() == 2:
        if X.dim() == 2:  # X is [N, d] (no batch) but graph_emb is batched → mismatch
            raise ValueError("graph_emb is batched but X is not")
        elif X.dim() == 3:  # [B, N, d]
            graph_emb_expanded = graph_emb.unsqueeze(1).expand(
                -1, X.size(1), -1
            )  # [B, N, d]

    else:
        raise ValueError("graph_emb must be 1D or 2D")

    return torch.cat([X, graph_emb_expanded], dim=-1)
