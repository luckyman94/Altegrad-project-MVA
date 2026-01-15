# -*- coding: utf-8 -*-

import torch
from typing import List


@torch.no_grad()
def fuse_soft_tokens_and_rag(
    llm,
    tokenizer,
    soft_tokens: torch.Tensor,
    retrieved_texts: List[List[str]],
    device: str,
):
    """
    soft_tokens: [B, S, D]
    retrieved_texts: List[B][K] (strings)
    """

    B = soft_tokens.size(0)
    emb_layer = llm.get_input_embeddings()
    dtype = emb_layer.weight.dtype

    # --------------------------------------------------
    # Flatten retrieved texts
    # --------------------------------------------------
    flat_texts = []
    for texts in retrieved_texts:
        flat_texts.append("\n".join(texts))

    # --------------------------------------------------
    # Tokenize retrieved docs
    # --------------------------------------------------
    tok = tokenizer(
        flat_texts,
        padding=True,
        truncation=True,
        return_tensors="pt",
    ).to(device)

    rag_emb = emb_layer(tok.input_ids).to(dtype=dtype)   # [B, L_rag, D]
    soft_tokens = soft_tokens.to(dtype=dtype)

    # --------------------------------------------------
    # Concatenate
    # --------------------------------------------------
    fused = torch.cat(
        [soft_tokens, rag_emb],
        dim=1,
    )

    return fused
