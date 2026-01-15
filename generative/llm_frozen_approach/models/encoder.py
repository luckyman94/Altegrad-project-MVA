import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import torch
import torch.nn as nn
from data_utils import x_map, e_map
from torch_geometric.nn import GPSConv, GINEConv, global_mean_pool, global_add_pool
import torch.nn.functional as F
from dataclasses import dataclass
import json
import torch
from pathlib import Path



@dataclass
class GraphEncoderConfig:
    hidden_dim: int = 256
    out_dim: int = 768
    num_layers: int = 4
    num_heads: int = 4
    dropout: float = 0.1
    attn_type: str = "multihead"
    pool: str = "mean"
    normalize_out: bool = True



class AtomEncoder(nn.Module):
    def __init__(self, hidden_dim, dropout = 0.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embs = nn.ModuleList([
        nn.Embedding(len(values), hidden_dim)
        for values in x_map.values()
        ])

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        for emb in self.embs:
            nn.init.normal_(emb.weight, mean=0.0, std=0.02)

    def forward(self, x) :
        if x.dim() != 2 or x.size(1) != len(self.embs):
            raise ValueError(f"AtomEncoder expected x shape (N,{len(self.embs)}), got {tuple(x.shape)}")
        x = x.long()

        h = 0
        for i, emb in enumerate(self.embs):
            h = h + emb(x[:, i])
        return self.dropout(h)
    

class BondEncoder(nn.Module):
    def __init__(self, hidden_dim, dropout = 0.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embs = nn.ModuleList([
        nn.Embedding(len(values), hidden_dim)
        for values in e_map.values()
        ])

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        for emb in self.embs:
            nn.init.normal_(emb.weight, mean=0.0, std=0.02)

    def forward(self, edge_attr) :
        if edge_attr.dim() != 2 or edge_attr.size(1) != len(self.embs):
            raise ValueError(f"BondEncoder expected edge_attr shape (E,{len(self.embs)}), got {tuple(edge_attr.shape)}")
        edge_attr = edge_attr.long()

        e = 0
        for i, emb in enumerate(self.embs):
            e = e + emb(edge_attr[:, i])
        return self.dropout(e)
    


class GPSBackbone(nn.Module):
    def __init__(
        self,
        hidden_dim,
        num_layers,
        num_heads,
        dropout,
        attn_type,
    ):
        super().__init__()
        self.layers = nn.ModuleList()

        for _ in range(num_layers):
            local_mlp = nn.Sequential(
                nn.Linear(hidden_dim, 2 * hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(2 * hidden_dim, hidden_dim),
            )
            local_conv = GINEConv(local_mlp, train_eps=True, edge_dim=hidden_dim)

            self.layers.append(
                GPSConv(
                    channels=hidden_dim,
                    conv=local_conv,
                    heads=num_heads,
                    dropout=dropout,
                    attn_type=attn_type,
                )
            )

    def forward(self, h_nodes: torch.Tensor, edge_index: torch.Tensor, batch_vec: torch.Tensor, h_edges: torch.Tensor) -> torch.Tensor:
        h = h_nodes
        e = h_edges
        for layer in self.layers:
            h = layer(h, edge_index, batch_vec, edge_attr=e)
        return h
    


class GraphEncoder(nn.Module):
    def __init__(self, cfg: GraphEncoderConfig):
        super().__init__()
        self.cfg = cfg

        self.atom_encoder = AtomEncoder(cfg.hidden_dim)
        self.bond_encoder = BondEncoder(cfg.hidden_dim, dropout=cfg.dropout)

        self.backbone = GPSBackbone(
            hidden_dim=cfg.hidden_dim,
            num_layers=cfg.num_layers,
            num_heads=cfg.num_heads,
            dropout=cfg.dropout,
            attn_type=cfg.attn_type,
        )

        if cfg.pool == "mean":
            self.pool = global_mean_pool
        elif cfg.pool == "add":
            self.pool = global_add_pool
        else:
            raise ValueError("cfg.pool must be 'mean' or 'add'")

        self.proj = nn.Sequential(
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim, cfg.out_dim),
        )

    def forward(self, batch) -> torch.Tensor:
        if not hasattr(batch, "batch") or batch.batch is None:
            batch.batch = torch.zeros(batch.x.size(0), dtype=torch.long, device=batch.x.device)

        h_nodes = self.atom_encoder(batch.x)         
        h_edges = self.bond_encoder(batch.edge_attr) 

        h_nodes = self.backbone(h_nodes, batch.edge_index, batch.batch, h_edges)

        g = self.pool(h_nodes, batch.batch)          

        z = self.proj(g)                             

        if self.cfg.normalize_out:
            z = F.normalize(z, dim=-1)

        return z


def load_graph_encoder_from_checkpoint(
    model_path: str,
    device: str,
    **kwargs  
):
    

    model_path = Path(model_path)
    model_dir = model_path.parent
    config_path = model_dir / "model_config.json"

    if not config_path.exists():
        raise FileNotFoundError(
            f"Missing GraphEncoder config file: {config_path}"
        )

    with open(config_path, "r") as f:
        cfg_dict = json.load(f)

    model_class = cfg_dict.get("model_class", "GraphEncoder")

    if model_class != "GraphEncoder":
        raise ValueError(
            f"Unsupported model_class '{model_class}', expected 'GraphEncoder'"
        )

    
    cfg = GraphEncoderConfig(
        hidden_dim=cfg_dict.get("hidden_dim", 256),
        out_dim=cfg_dict.get("out_dim", 768),
        num_layers=cfg_dict.get("num_layers", 4),
        num_heads=cfg_dict.get("num_heads", 4),
        dropout=cfg_dict.get("dropout", 0.1),
        attn_type=cfg_dict.get("attn_type", "multihead"),
        pool=cfg_dict.get("pool", "mean"),
        normalize_out=cfg_dict.get("normalize_out", True),
    )

    
    gnn = GraphEncoder(cfg).to(device)

    state = torch.load(model_path, map_location=device)
    gnn.load_state_dict(state)

    gnn.eval()

    return gnn


if __name__ == "__main__":
    cfg = GraphEncoderConfig(hidden_dim=256, out_dim=768, num_layers=4, num_heads=4, dropout=0.1)
    model = GraphEncoder(cfg)
    print(GraphEncoderConfig.__annotations__)
    print(model)







