#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import sys
from pathlib import Path
from typing import List

import torch
from torch_geometric.data import Batch
from tqdm import tqdm
import pandas as pd

from peft import PeftModel

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

# ===================== Imports projet =====================
from data_utils import PreprocessedGraphDataset
from sft_llm_mapper.models.encoder import GraphEncoder, GraphEncoderConfig
from sft_llm_mapper.models.mapper import LinearMapper
from sft_llm_mapper.models.llm_factory import load_llm
from sft_llm_mapper.rag.fusion import fuse_soft_tokens_and_rag


# ======================================================
# Dummy Retriever (baseline)
# ======================================================
class DummyRetriever:
    """
    Baseline retriever : retourne toujours les mêmes textes.
    À remplacer par FAISS plus tard.
    """

    def __init__(self, texts: List[str]):
        self.texts = texts

    def search(self, query_emb: torch.Tensor, k: int = 3):
        return [[t for t in self.texts[:k]] for _ in range(query_emb.size(0))]


# ======================================================
# Generation (RAG)
# ======================================================
@torch.no_grad()
def generate_batch_rag(
    graphs,
    graph_encoder,
    mapper,
    llm,
    tokenizer,
    retriever,
    device,
    max_new_tokens,
    num_beams,
):
    graphs = graphs.to(device)

    # -------- Graph → soft tokens --------
    z_graph = graph_encoder(graphs)      # [B, Dg]
    soft = mapper(z_graph)               # [B, S, Dllm]

    # -------- Retrieve --------
    retrieved_texts = retriever.search(z_graph, k=3)

    # -------- Fuse soft + RAG --------
    inputs_embeds = fuse_soft_tokens_and_rag(
        llm=llm,
        tokenizer=tokenizer,
        soft_tokens=soft,
        retrieved_texts=retrieved_texts,
        device=device,
    )

    attention_mask = torch.ones(
        inputs_embeds.shape[:2],
        device=device,
        dtype=torch.long,
    )

    outputs = llm.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        num_beams=num_beams,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    return tokenizer.batch_decode(outputs, skip_special_tokens=True)


# ======================================================
# Main
# ======================================================
def main():
    p = argparse.ArgumentParser("Graph → Text inference with RAG")

    p.add_argument("--test_data", required=True)
    p.add_argument("--encoder_ckpt", required=True)
    p.add_argument("--mapper_ckpt", required=True)

    p.add_argument("--llm", choices=["gpt2", "biogpt"], required=True)
    p.add_argument("--lora_path", required=True, help="Path to trained LoRA adapter")

    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--max_new_tokens", type=int, default=128)
    p.add_argument("--num_beams", type=int, default=1)

    p.add_argument("--device", default="cuda")
    p.add_argument("--out_csv", default="submission.csv")

    args = p.parse_args()
    device = args.device if torch.cuda.is_available() else "cpu"

    # --------------------------------------------------
    # Dataset
    # --------------------------------------------------
    test_ds = PreprocessedGraphDataset(args.test_data)
    test_loader = torch.utils.data.DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda x: x,
    )

    # --------------------------------------------------
    # Graph encoder
    # --------------------------------------------------
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

    graph_encoder = GraphEncoder(cfg).to(device)
    graph_encoder.load_state_dict(enc_ckpt["graph_encoder_state_dict"])
    graph_encoder.eval()

    # --------------------------------------------------
    # Mapper
    # --------------------------------------------------
    map_ckpt = torch.load(args.mapper_ckpt, map_location=device)

    mapper = LinearMapper(
        dim_graph=map_ckpt["dim_graph"],
        dim_llm=map_ckpt["dim_llm"],
        num_soft_tokens=map_ckpt["num_soft_tokens"],
    ).to(device)

    mapper.load_state_dict(map_ckpt["mapper_state"])
    mapper.eval()

    # --------------------------------------------------
    # LLM + LoRA (CORRECT)
    # --------------------------------------------------
    llm, tokenizer, _ = load_llm(
        llm_name=args.llm,
        device=device,
        use_lora=False,   # ⬅️ important
    )

    llm = PeftModel.from_pretrained(
        llm,
        args.lora_path,
    ).to(device)

    llm.eval()

    # --------------------------------------------------
    # Retriever
    # --------------------------------------------------
    retriever = DummyRetriever(
        texts=[
            "This molecule contains aromatic rings.",
            "Functional groups include hydroxyl and amine.",
            "Likely exhibits polar behavior.",
        ]
    )

    # --------------------------------------------------
    # Inference
    # --------------------------------------------------
    rows = []

    for batch in tqdm(test_loader, desc="Inference (RAG)"):
        graphs = Batch.from_data_list(batch)
        ids = [g.id for g in batch]

        texts = generate_batch_rag(
            graphs,
            graph_encoder,
            mapper,
            llm,
            tokenizer,
            retriever,
            device,
            args.max_new_tokens,
            args.num_beams,
        )

        for gid, txt in zip(ids, texts):
            rows.append({"ID": gid, "description": txt})

    pd.DataFrame(rows).to_csv(args.out_csv, index=False)
    print(f"Saved RAG predictions to {args.out_csv}")


if __name__ == "__main__":
    main()
