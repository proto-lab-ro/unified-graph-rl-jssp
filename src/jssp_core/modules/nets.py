import torch
from torch import nn


# from torch_geometric.data import nn


class CriticValueHead(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.value_head = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

        for m in self.value_head:
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.value_head(x)


class CriticAgvValueHead(nn.Module):
    def __init__(self, agv_in_dim: int):
        super().__init__()
        self.agv_value_head = nn.Sequential(
            nn.Linear(agv_in_dim, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 1),
        )

        for m in self.agv_value_head:
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.agv_value_head(x)


class CriticValueHead_Small_16(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.value_head = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

        for m in self.value_head:
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.value_head(x)


class ActorPolicyHead(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.actor_policy_head = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )
        # Initialize weights
        for m in self.actor_policy_head:
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.actor_policy_head(x)


class SmallActorPolicyHead(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.actor_policy_head = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.Tanh(),
            nn.Linear(16, 16),
            nn.Tanh(),
            nn.Linear(16, 1),
        )
        # Initialize weights
        for m in self.actor_policy_head:
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.actor_policy_head(x)
