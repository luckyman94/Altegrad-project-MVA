import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from torch_geometric.data import Batch
from torch_geometric.nn import GCNConv,GINEConv, global_add_pool
from data_utils import x_map, e_map


class AtomEncoder(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.feature_names = list(x_map.keys())
        self.embeddings = nn.ModuleDict({
            name: nn.Embedding(len(values), hidden_dim)
            for name, values in x_map.items()
        })
    def forward(self, x):
        
        x = x.long()

        h = 0
        for i, name in enumerate(self.feature_names):
            emb = self.embeddings[name]
            h = h + emb(x[:, i].clamp(0, emb.num_embeddings - 1))

        return h


class EdgeEncoder(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()

        self.feature_names = list(e_map.keys())

        self.embeddings = nn.ModuleDict({
            name: nn.Embedding(len(values), hidden_dim)
            for name, values in e_map.items()
        })

    def forward(self, edge_attr):
        edge_attr = edge_attr.long()

        h = 0
        for i, name in enumerate(self.feature_names):
            emb = self.embeddings[name]
            h = h + emb(edge_attr[:, i].clamp(0, emb.num_embeddings - 1))

        return h


class MolGNN(nn.Module):
    def __init__(self, hidden=128, out_dim=256, layers=3):
        super().__init__()

        
        self.atom_encoder = AtomEncoder(hidden)
        self.edge_encoder = EdgeEncoder(hidden)

        self.convs = nn.ModuleList()
        for _ in range(layers):
            mlp = nn.Sequential(
                nn.Linear(hidden,hidden),
                nn.ReLU(),
                nn.Linear(hidden,hidden)
            )
            self.convs.append(GINEConv(mlp))

        self.proj = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim)
        )

    def forward(self, batch: Batch):
        h = self.atom_encoder(batch.x)
        edge_attr = self.edge_encoder(batch.edge_attr)
        
        for conv in self.convs:
            h = conv(h, batch.edge_index,edge_attr=edge_attr)
            h = F.relu(h)
        g = global_add_pool(h, batch.batch)
        g = self.proj(g)
        g = F.normalize(g, dim=-1)
        return g
