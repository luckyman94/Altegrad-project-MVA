#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Naive Graph-RAG inference
- No FAISS
- Retrieval = brute-force cosine similarity in graph embedding space
- Context = top-k train descriptions
"""

import argparse
import sys
from pathlib import Path
from typing import List

import torch
import torch.nn.functional as F
from torch_geometric.data import Batch
from tqdm import tqdm
import pandas as pd

from peft import PeftModel

# --------------------------------------------------
# Path setup
# --------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

# --------------------------------------------------
# Project imports
# --------------------------------------------------
from data_utils import PreprocessedGraphDataset
from sft_llm_mapper.models.encoder import GraphEncoder, GraphEncoderConfig
from sft_llm_mapper.models.mapper import LinearMapper
from sft_llm_mapper.models.llm_factory import load_llm


# ======================================================
# Naive in-memory graph retriever
# ======================================================
class NaiveGraphRetriever:
    """
    Stores all train graph embeddings in memory
    Retrieval by cosine similarity
    """

    def __init__(
        self,
        graph_encoder,
        train_ds,
        device,
    ):
        self.device = device
        self.graph_encoder = graph_encoder.to(device).eval()

        self.embs = []
        self.texts = []

        print("🔎 Encoding train graphs for naive RAG")

        with torch.no_grad():
            for g in tqdm(train_ds, desc="Encoding train graphs"):
                g = g.to(device)
                z = self.graph_encoder(Batch.from_data_list([g]))  # [1, D]
                z = F.normalize(z, dim=-1)
                self.embs.append(z.cpu())
                self.texts.append(g.description)

        self.embs = torch.cat(self.embs, dim=0)  # [N, D]
        print(f"✅ Stored {len(self.texts)} train descriptions")

    def search(self, query_emb: torch.Tensor, k: int = 3):
        """
        query_emb: [B, D]
        returns: List[List[str]]
        """
        query_emb = F.normalize(query_emb, dim=-1)
        sims = query_emb @ self.embs.T  # [B, N]

        topk = torch.topk(sims, k=k, dim=1).indices

        retrieved = []
        for idxs in topk:
            retrieved.append([self.texts[i] for i in idxs.tolist()])

        return retrieved


# ======================================================
# Fusion: soft tokens + optional RAG context
# ======================================================
@torch.no_grad()
def fuse_soft_tokens_and_rag(
    llm,
    tokenizer,
    soft_tokens,
    retrieved_texts,
    device,
):
    emb_layer = llm.get_input_embeddings()
    dtype = emb_layer.weight.dtype

    soft_tokens = soft_tokens.to(dtype)
    batch_inputs = []

    for i in range(soft_tokens.size(0)):
        soft_i = soft_tokens[i]  # [S, D]

        texts = retrieved_texts[i]
        has_context = texts is not None and len(texts) > 0

        if has_context:
            context = "\n".join(t for t in texts if t.strip() != "")
            prompt = (
                "Context:\n"
                f"{context}\n\n"
                "Task: Describe the molecule."
            )
        else:
            prompt = "Describe the molecule."

        tok = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            padding=False,
        ).to(device)

        emb_prompt = emb_layer(tok.input_ids).squeeze(0).to(dtype)

        fused = torch.cat([soft_i, emb_prompt], dim=0)
        batch_inputs.append(fused)

    inputs_embeds = torch.nn.utils.rnn.pad_sequence(
        batch_inputs,
        batch_first=True,
    )

    attention_mask = torch.ones(
        inputs_embeds.shape[:2],
        device=device,
        dtype=torch.long,
    )

    return inputs_embeds, attention_mask


# ======================================================
# Generation
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

    # Graph → soft tokens
    z_graph = graph_encoder(graphs)          # [B, Dg]
    soft = mapper(z_graph)                   # [B, S, Dllm]

    # Retrieve train descriptions
    retrieved_texts = retriever.search(z_graph, k=3)

    # Fuse
    inputs_embeds, attention_mask = fuse_soft_tokens_and_rag(
        llm,
        tokenizer,
        soft,
        retrieved_texts,
        device,
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
    p = argparse.ArgumentParser("Graph → Text inference (NAIVE RAG)")

    p.add_argument("--train_data", required=True)
    p.add_argument("--test_data", required=True)

    p.add_argument("--encoder_ckpt", required=True)
    p.add_argument("--mapper_ckpt", required=True)

    p.add_argument("--llm", choices=["gpt2", "biogpt"], required=True)
    p.add_argument("--lora_path", required=True)

    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--max_new_tokens", type=int, default=128)
    p.add_argument("--num_beams", type=int, default=1)

    p.add_argument("--device", default="cuda")
    p.add_argument("--out_csv", default="submission.csv")

    args = p.parse_args()
    device = args.device if torch.cuda.is_available() else "cpu"

    # --------------------------------------------------
    # Datasets
    # --------------------------------------------------
    train_ds = PreprocessedGraphDataset(args.train_data)
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
    # LLM + LoRA
    # --------------------------------------------------
    llm, tokenizer, _ = load_llm(
        llm_name=args.llm,
        device=device,
        use_lora=False,
    )

    llm = PeftModel.from_pretrained(llm, args.lora_path).to(device)
    llm.eval()

    # --------------------------------------------------
    # Naive retriever
    # --------------------------------------------------
    retriever = NaiveGraphRetriever(
        graph_encoder=graph_encoder,
        train_ds=train_ds,
        device=device,
    )

    # --------------------------------------------------
    # Inference
    # --------------------------------------------------
    rows = []

    for batch in tqdm(test_loader, desc="Inference (Naive RAG)"):
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
    print(f"✅ Saved predictions to {args.out_csv}")


if __name__ == "__main__":
    main()
