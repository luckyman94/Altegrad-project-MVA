import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from torch_geometric.data import Batch
from torch_geometric.nn import GCNConv,GINEConv, global_add_pool


class AtomEncoder(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.emb_atomic_num = nn.Embedding(119, hidden_dim)  # atom numbers up to 118
        self.emb_chirality = nn.Embedding(9, hidden_dim)    # 9 types of chirality
        self.emb_degree = nn.Embedding(11, hidden_dim)      # 0-10
        self.emb_charge = nn.Embedding(12, hidden_dim)      # -5 to +6
        self.emb_num_hs = nn.Embedding(9, hidden_dim)      # 0-8
        self.emb_rad_el = nn.Embedding(5, hidden_dim)      # 0-4
        self.emb_hybrid = nn.Embedding(8, hidden_dim)      
        self.emb_aromatic = nn.Embedding(2, hidden_dim)     # True/False
        self.emb_ring = nn.Embedding(2, hidden_dim)  # True/False
    def forward(self, x):
        # Ensure integer indices for embedding lookups
        x = x.long()

        atomic_num = self.emb_atomic_num(x[:, 0].clamp(0, 118))
        chirality = self.emb_chirality(x[:, 1].clamp(0, 8))
        degree = self.emb_degree(x[:, 2].clamp(0, 10))
        charge = self.emb_charge(x[:, 3].clamp(0, 11))
        num_hs = self.emb_num_hs(x[:, 4].clamp(0, 8))
        rad_el = self.emb_rad_el(x[:, 5].clamp(0, 4))
        hybrid = self.emb_hybrid(x[:, 6].clamp(0, 7))
        aromatic = self.emb_aromatic(x[:, 7].clamp(0, 1))
        ring = self.emb_ring(x[:, 8].clamp(0, 1))
        
        h = (atomic_num + chirality + degree + charge +
             num_hs + rad_el + hybrid + aromatic + ring)
        return h


class EdgeEncoder(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.emb_bond_type = nn.Embedding(23, hidden_dim)
        self.emb_stereo = nn.Embedding(6,hidden_dim)
        self.emb_conjugated = nn.Embedding(2,hidden_dim)
    
    def forward(self, edge_attr):
        return self.emb_bond_type(edge_attr[:, 0]) + \
               self.emb_stereo(edge_attr[:, 1]) + \
               self.emb_conjugated(edge_attr[:, 2])


# =========================================================
# MODEL: GNN to encode graphs (add edge features)
# =========================================================
class MolGNN(nn.Module):
    def __init__(self, hidden=128, out_dim=256, layers=3):
        super().__init__()

        # Use a single learnable embedding for all nodes (no node features)
        #self.node_init = nn.Parameter(torch.randn(hidden))
        # add learnable embedding for node features
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
        # Initialize all nodes with the same learnable embedding
        # num_nodes = batch.x.size(0)
        # h = self.node_init.unsqueeze(0).expand(num_nodes, -1)
        h = self.atom_encoder(batch.x)
        edge_attr = self.edge_encoder(batch.edge_attr)
        
        for conv in self.convs:
            h = conv(h, batch.edge_index,edge_attr=edge_attr)
            h = F.relu(h)
        g = global_add_pool(h, batch.batch)
        g = self.proj(g)
        g = F.normalize(g, dim=-1)
        return g
