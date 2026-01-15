import torch
import torch.nn as nn
from torch_geometric.nn import GINEConv, global_mean_pool
from pathlib import Path

import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))
from constants import LATENT_TOKENS, D_MODEL


class GraphToLatent(nn.Module):
    def __init__(self):
        super().__init__()
        H = 256

        self.node_proj = nn.Linear(9, H)

        self.convs = nn.ModuleList([
            GINEConv(
                nn.Sequential(
                    nn.Linear(H, H),
                    nn.ReLU(),
                    nn.Linear(H, H)
                ),
                edge_dim=3
            )
            for _ in range(3)
        ])

        self.norms = nn.ModuleList([nn.LayerNorm(H) for _ in range(3)])

        self.readout = nn.Sequential(
            nn.Linear(H, 1024),
            nn.ReLU(),
            nn.Linear(1024, LATENT_TOKENS * D_MODEL)
        )

    def forward(self, batch):
        x = self.node_proj(batch.x.float())

        for conv, norm in zip(self.convs, self.norms):
            x = norm(x + conv(x, batch.edge_index, edge_attr=batch.edge_attr.float()))

        g = global_mean_pool(x, batch.batch)
        return self.readout(g).view(-1, LATENT_TOKENS, D_MODEL)