#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import sys
from pathlib import Path
from typing import List

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Batch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

# ========= Project imports =========
from data_utils import PreprocessedGraphDataset
from sft_llm_mapper.models.encoder import GraphEncoder, GraphEncoderConfig
from sft_llm_mapper.models.mapper import LinearMapper
from sft_llm_mapper.models.llm_factory import load_llm


# ======================================================
# Prompting
# ======================================================
SYSTEM_PROMPT = (
    "You are an expert chemist. "
    "Describe the molecule concisely, focusing on its functional groups and properties."
)

USER_TEMPLATE = "Molecule description:"



# ======================================================
# Dataset
# ======================================================
class GraphTextSFTDataset(Dataset):
    def __init__(self, base_ds: PreprocessedGraphDataset):
        self.ds = base_ds

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        g = self.ds[idx]
        return {
            "graph": g,
            "prompt": SYSTEM_PROMPT + "\n\n" + USER_TEMPLATE,
            "target": g.description,
        }



def collate_sft(batch):
    return {
        "graphs": Batch.from_data_list([b["graph"] for b in batch]),
        "prompts": [b["prompt"] for b in batch],
        "targets": [b["target"] for b in batch],
    }


# ======================================================
# Utilities
# ======================================================
def build_inputs_with_soft_tokens(
    llm,
    tokenizer,
    soft_tokens: torch.Tensor,
    prompts: List[str],
    targets: List[str],
    device: str,
):
    tok_prompt = tokenizer(
        prompts,
        padding=True,
        return_tensors="pt",
    ).to(device)

    tok_target = tokenizer(
        targets,
        padding=True,
        return_tensors="pt",
    ).to(device)

    emb_prompt = llm.get_input_embeddings()(tok_prompt.input_ids)
    emb_target = llm.get_input_embeddings()(tok_target.input_ids)

    inputs_embeds = torch.cat(
        [soft_tokens, emb_prompt, emb_target[:, :-1]],
        dim=1,
    )

    labels = torch.cat(
        [
            torch.full(
                (soft_tokens.size(0), soft_tokens.size(1) + emb_prompt.size(1)),
                -100,
                device=device,
            ),
            tok_target.input_ids,
        ],
        dim=1,
    )

    return inputs_embeds, labels


# ======================================================
# Stage-2 training
# ======================================================
def train_stage2(
    graph_encoder,
    mapper,
    llm,
    tokenizer,
    train_ds,
    val_ds,
    device,
    epochs,
    batch_size,
    lr_mapper,
    lr_lora,
    out_dir,
):
    os.makedirs(out_dir, exist_ok=True)

    graph_encoder.eval().to(device)
    for p in graph_encoder.parameters():
        p.requires_grad = False

    mapper.train().to(device)
    llm.train().to(device)

    optimizer = torch.optim.AdamW(
        [
            {"params": mapper.parameters(), "lr": lr_mapper},
            {
                "params": [p for p in llm.parameters() if p.requires_grad],
                "lr": lr_lora,
            },
        ],
        weight_decay=1e-4,
    )

    train_loader = DataLoader(
        GraphTextSFTDataset(train_ds),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_sft,
    )

    val_loader = DataLoader(
        GraphTextSFTDataset(val_ds),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_sft,
    )

    best_val = float("inf")

    for epoch in range(1, epochs + 1):

        # -------- TRAIN --------
        llm.train()
        mapper.train()
        tr_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Stage2 {epoch} [train]"):
            optimizer.zero_grad()

            graphs = batch["graphs"].to(device)

            with torch.no_grad():
                z_graph = graph_encoder(graphs)

            soft = mapper(z_graph)

            inputs_embeds, labels = build_inputs_with_soft_tokens(
                llm,
                tokenizer,
                soft,
                batch["prompts"],
                batch["targets"],
                device,
            )

            out = llm(
                inputs_embeds=inputs_embeds,
                labels=labels,
            )

            loss = out.loss
            loss.backward()
            optimizer.step()

            tr_loss += loss.item()

        tr_loss /= len(train_loader)

        # -------- VAL --------
        llm.eval()
        mapper.eval()
        va_loss = 0.0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Stage2 {epoch} [val]"):
                graphs = batch["graphs"].to(device)
                z_graph = graph_encoder(graphs)
                soft = mapper(z_graph)

                inputs_embeds, labels = build_inputs_with_soft_tokens(
                    llm,
                    tokenizer,
                    soft,
                    batch["prompts"],
                    batch["targets"],
                    device,
                )

                out = llm(inputs_embeds=inputs_embeds, labels=labels)
                va_loss += out.loss.item()

        va_loss /= len(val_loader)

        print(
            f"[Stage2] epoch={epoch} "
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
                os.path.join(out_dir, "best_mapper.pt"),
            )

            llm.save_pretrained(os.path.join(out_dir, "lora"))
            tokenizer.save_pretrained(os.path.join(out_dir, "lora"))

            print(f"✓ Saved best Stage-2 model")

    return best_val


# ======================================================
# Main
# ======================================================
def main():
    p = argparse.ArgumentParser("Stage-2 SFT (Graph → Text)")
    p.add_argument("--train_data", required=True)
    p.add_argument("--val_data", required=True)
    p.add_argument("--encoder_ckpt", required=True)
    p.add_argument("--mapper_ckpt", required=True)
    p.add_argument("--llm", choices=["gpt2", "biogpt"], required=True)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr_mapper", type=float, default=3e-4)
    p.add_argument("--lr_lora", type=float, default=1e-4)
    p.add_argument("--out_dir", default="checkpoints/stage2")
    p.add_argument("--device", default="cuda")

    args = p.parse_args()
    device = args.device if torch.cuda.is_available() else "cpu"

    # -------- Encoder --------
    enc_ckpt = torch.load(args.encoder_ckpt, map_location=device)
    cfg = GraphEncoderConfig(
        hidden_dim=enc_ckpt["gnn_hidden_dim"],
        out_dim=enc_ckpt["gnn_out_dim"],
        num_layers=enc_ckpt["num_layers"],
        num_heads=enc_ckpt["num_heads"],
        dropout=enc_ckpt["dropout"],
        attn_type=enc_ckpt["attn_type"],
        pool=enc_ckpt["pool"],
        normalize_out=enc_ckpt["normalize_out"],
    )
    graph_encoder = GraphEncoder(cfg)
    graph_encoder.load_state_dict(enc_ckpt["graph_encoder_state_dict"])

    # -------- Mapper --------
    mapper_ckpt = torch.load(args.mapper_ckpt, map_location=device)
    mapper = LinearMapper(
        dim_graph=mapper_ckpt["dim_graph"],
        dim_llm=mapper_ckpt["dim_llm"],
        num_soft_tokens=mapper_ckpt["num_soft_tokens"],
    )
    mapper.load_state_dict(mapper_ckpt["mapper_state"])

    # -------- LLM + LoRA --------
    llm, tokenizer, _ = load_llm(
        llm_name=args.llm,
        device=device,
        use_lora=True,
    )

    train_ds = PreprocessedGraphDataset(args.train_data)
    val_ds = PreprocessedGraphDataset(args.val_data)

    train_stage2(
        graph_encoder,
        mapper,
        llm,
        tokenizer,
        train_ds,
        val_ds,
        device,
        args.epochs,
        args.batch_size,
        args.lr_mapper,
        args.lr_lora,
        args.out_dir,
    )


if __name__ == "__main__":
    main()
