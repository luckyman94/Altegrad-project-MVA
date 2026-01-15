import torch
from torch.utils.data import DataLoader
from torch_geometric.data import Batch
from pathlib import Path
import sys
import argparse
from tqdm import tqdm
import os
from transformers import get_linear_schedule_with_warmup
from torch.cuda.amp import autocast, GradScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from data_utils import PreprocessedGraphDataset
from dataset.dataset import GraphTextDataset
from models.encoder import GraphEncoder, GraphEncoderConfig
from models.mapper import LinearMapper
from models.biot5 import load_biot5, get_llm_dim

def build_encoder_prompt(tokenizer, device):
    text = "Describe the following molecule:"
    ids = tokenizer(
        text,
        return_tensors="pt",
        add_special_tokens=False,
    ).input_ids.to(device)
    return ids

def parse_args():
    p = argparse.ArgumentParser("Graph → BioT5 (frozen)")

    p.add_argument("--train_graphs", type=str, required=True)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--max_text_len", type=int, default=128)
    p.add_argument("--num_soft_tokens", type=int, default=4)
    p.add_argument("--checkpoint_path", type=str, default="checkpoints_biot5")

    return p.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_graphs = PreprocessedGraphDataset(args.train_graphs)
    val_graphs = PreprocessedGraphDataset(
        args.train_graphs.replace("train", "validation")
    )

    train_ds = GraphTextDataset(train_graphs, max_length=args.max_text_len)
    val_ds = GraphTextDataset(val_graphs, max_length=args.max_text_len)

    collate_fn = lambda x: {
        "graph": [item["graph"] for item in x],
        "input_ids": torch.stack([item["input_ids"] for item in x]),
    }

    train_loader = DataLoader(train_ds, args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, args.batch_size, shuffle=False, collate_fn=collate_fn)

    llm, tokenizer = load_biot5(device)
    llm_dim = get_llm_dim(llm)

    encoder_prompt_ids = build_encoder_prompt(tokenizer, device)
    encoder_prompt_embeds = llm.encoder.embed_tokens(encoder_prompt_ids)

    cfg = GraphEncoderConfig(
        hidden_dim=args.hidden_dim,
        out_dim=llm_dim,
        num_layers=4,
        num_heads=4,
        dropout=0.1,
    )
    graph_encoder = GraphEncoder(cfg).to(device)

    mapper = LinearMapper(
        dim_graph=llm_dim,
        dim_llm=llm_dim,
        num_soft_tokens=args.num_soft_tokens,
    ).to(device)

    optimizer = torch.optim.AdamW(
        [
            {"params": graph_encoder.parameters(), "lr": 5e-5},
            {"params": mapper.parameters(), "lr": 3e-4},
        ]
    )

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * args.epochs * len(train_loader)),
        num_training_steps=args.epochs * len(train_loader),
    )

    scaler = GradScaler()
    best_val = float("inf")

    for epoch in range(args.epochs):
        graph_encoder.train()
        mapper.train()
        train_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1} [train]"):
            optimizer.zero_grad(set_to_none=True)

            graph_batch = Batch.from_data_list(batch["graph"]).to(device)
            labels = batch["input_ids"].to(device)

            with autocast():
                g = graph_encoder(graph_batch)
                soft_prompt = mapper(g)  # (B, S, D)

                B = labels.size(0)
                enc_prompt = encoder_prompt_embeds.repeat(B, 1, 1)

                encoder_inputs = torch.cat(
                    [soft_prompt, enc_prompt],
                    dim=1,
                )

                outputs = llm(
                    inputs_embeds=encoder_inputs,
                    labels=labels,
                )
                loss = outputs.loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        graph_encoder.eval()
        mapper.eval()
        val_loss = 0.0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1} [val]"):
                graph_batch = Batch.from_data_list(batch["graph"]).to(device)
                labels = batch["input_ids"].to(device)

                with autocast():
                    g = graph_encoder(graph_batch)
                    soft_prompt = mapper(g)

                    B = labels.size(0)
                    enc_prompt = encoder_prompt_embeds.repeat(B, 1, 1)

                    encoder_inputs = torch.cat(
                        [soft_prompt, enc_prompt],
                        dim=1,
                    )

                    outputs = llm(
                        inputs_embeds=encoder_inputs,
                        labels=labels,
                    )
                    val_loss += outputs.loss.item()

        val_loss /= len(val_loader)

        print(
            f"Epoch {epoch+1}/{args.epochs} | "
            f"train={train_loss:.4f} | val={val_loss:.4f}"
        )

        if val_loss < best_val:
            best_val = val_loss
            os.makedirs(args.checkpoint_path, exist_ok=True)
            torch.save(
                {
                    "graph_encoder": graph_encoder.state_dict(),
                    "mapper": mapper.state_dict(),
                    "gnn_hidden_dim": cfg.hidden_dim,
                    "gnn_out_dim": cfg.out_dim,
                    "num_soft_tokens": args.num_soft_tokens,
                },
                os.path.join(args.checkpoint_path, "best_graph2biot5.pt"),
            )

    print("Training finished.")
