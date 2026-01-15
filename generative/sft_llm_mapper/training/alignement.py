#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from torch_geometric.data import Batch

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

# ===== Project imports =====
from data_utils import PreprocessedGraphDataset
from sft_llm_mapper.models.encoder import load_graph_encoder_from_checkpoint
from sft_llm_mapper.models.mapper import LinearMapper
from sft_llm_mapper.models.llm_factory import load_llm_embedder
from sft_llm_mapper.models.encoder import GraphEncoder, GraphEncoderConfig


# ======================================================
# Dataset
# ======================================================
class GraphTextAlignmentDataset(Dataset):
    def __init__(self, base: PreprocessedGraphDataset):
        self.base = base

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        g = self.base[idx]
        if not hasattr(g, "description"):
            raise AttributeError("Graph must have a `description` attribute")
        return g, g.description


def collate_stage1(batch: List[Tuple]):
    graphs = Batch.from_data_list([g for g, _ in batch])
    texts = [t for _, t in batch]
    return graphs, texts

def load_graph_encoder_simple(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)

    cfg = GraphEncoderConfig(
        hidden_dim=ckpt["gnn_hidden_dim"],
        out_dim=ckpt["gnn_out_dim"],
        num_layers=ckpt["num_layers"],
        num_heads=ckpt["num_heads"],
        dropout=ckpt["dropout"],
        attn_type=ckpt["attn_type"],
        pool=ckpt["pool"],
        normalize_out=ckpt["normalize_out"],
    )

    model = GraphEncoder(cfg).to(device)
    model.load_state_dict(ckpt["graph_encoder_state_dict"])
    model.eval()

    return model


# ======================================================
# Training loop (Stage-1 alignment)
# ======================================================
def train_stage1(
    graph_encoder,
    mapper,
    llm_embedder,
    train_ds,
    val_ds,
    device,
    epochs,
    batch_size,
    lr,
    out_ckpt,
):
    graph_encoder.eval().to(device)
    for p in graph_encoder.parameters():
        p.requires_grad = False

    mapper.train().to(device)
    for p in mapper.parameters():
        p.requires_grad = True

    optimizer = torch.optim.AdamW(
        mapper.parameters(),
        lr=lr,
        weight_decay=1e-4,
    )
    criterion = nn.MSELoss()

    train_loader = DataLoader(
        GraphTextAlignmentDataset(train_ds),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_stage1,
    )
    val_loader = DataLoader(
        GraphTextAlignmentDataset(val_ds),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_stage1,
    )

    best_val = float("inf")

    for epoch in range(1, epochs + 1):
        # -------- TRAIN --------
        mapper.train()
        tr_loss = 0.0

        for graphs, texts in tqdm(train_loader, desc=f"Stage1 {epoch} [train]"):
            graphs = graphs.to(device)
            optimizer.zero_grad(set_to_none=True)

            with torch.no_grad():
                g_emb = graph_encoder(graphs)        # [B, dim_graph]

            soft = mapper(g_emb).mean(dim=1)         # [B, dim_llm]

            with torch.no_grad():
                tgt = llm_embedder.encode(texts)     # [B, dim_llm]

            loss = criterion(soft, tgt)
            loss.backward()
            optimizer.step()

            tr_loss += loss.item()

        tr_loss /= max(1, len(train_loader))

        # -------- VAL --------
        mapper.eval()
        va_loss = 0.0

        with torch.no_grad():
            for graphs, texts in tqdm(val_loader, desc=f"Stage1 {epoch} [val]"):
                graphs = graphs.to(device)
                g_emb = graph_encoder(graphs)
                soft = mapper(g_emb).mean(dim=1)
                tgt = llm_embedder.encode(texts)
                va_loss += criterion(soft, tgt).item()

        va_loss /= max(1, len(val_loader))

        print(
            f"[Stage1] epoch={epoch} "
            f"train={tr_loss:.4f} val={va_loss:.4f}"
        )

        if va_loss < best_val:
            best_val = va_loss
            torch.save(
                {
                    "mapper_state": mapper.state_dict(),
                    "epoch": epoch,
                    "val_loss": va_loss,
                },
                out_ckpt,
            )
            print(f"✓ Saved best mapper → {out_ckpt}")

    return best_val


# ======================================================
# Argparse
# ======================================================
def parse_args():
    p = argparse.ArgumentParser("Stage-1 Graph → LLM Alignment")

    p.add_argument("--llm", type=str, choices=["gpt2", "biogpt"], required=True)
    p.add_argument("--graph_ckpt", type=str, required=True)
    p.add_argument("--train_data", type=str, required=True)
    p.add_argument("--val_data", type=str, required=True)

    p.add_argument("--num_soft_tokens", type=int, default=16)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-4)

    p.add_argument("--out_ckpt", type=str, default="stage1_mapper.pt")
    p.add_argument("--device", type=str, default="cuda")

    return p.parse_args()


# ======================================================
# Main
# ======================================================
def main():
    args = parse_args()
    device = args.device if torch.cuda.is_available() else "cpu"

    # -------- Load graph encoder --------
    graph_encoder = load_graph_encoder_from_checkpoint(
        model_path=args.graph_ckpt,
        device=device,
    )

    dim_graph = graph_encoder.cfg.out_dim

    # -------- Load LLM embedder (Stage-1 ONLY) --------
    llm_embedder = load_llm_embedder(
        llm_name=args.llm,
        device=device,
    )

    # -------- Mapper --------
    mapper = LinearMapper(
        dim_graph=dim_graph,
        dim_llm=llm_embedder.hidden_size,
        num_soft_tokens=args.num_soft_tokens,
    )

    # -------- Datasets --------
    train_ds = PreprocessedGraphDataset(args.train_data)
    val_ds = PreprocessedGraphDataset(args.val_data)

    # -------- Train --------
    train_stage1(
        graph_encoder=graph_encoder,
        mapper=mapper,
        llm_embedder=llm_embedder,
        train_ds=train_ds,
        val_ds=val_ds,
        device=device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        out_ckpt=args.out_ckpt,
    )


if __name__ == "__main__":
    main()
