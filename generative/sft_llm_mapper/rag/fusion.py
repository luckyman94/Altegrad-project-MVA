# -*- coding: utf-8 -*-

import torch
from typing import List


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

    rag_flat = [
        " ".join(texts) for texts in retrieved_texts
    ]

    rag_inputs = tokenizer(
        rag_flat,
        padding=True,
        truncation=True,
        return_tensors="pt",
    ).to(device)

    rag_embeds = emb_layer(rag_inputs.input_ids).to(dtype)

    # 🔥 PROMPT FINAL (OBLIGATOIRE)
    prompt_inputs = tokenizer(
        ["Describe the molecule."] * soft_tokens.size(0),
        return_tensors="pt",
    ).to(device)

    prompt_embeds = emb_layer(prompt_inputs.input_ids).to(dtype)

    return torch.cat(
        [soft_tokens, rag_embeds, prompt_embeds],
        dim=1,
    )
