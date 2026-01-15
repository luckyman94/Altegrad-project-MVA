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
    model_dtype = emb_layer.weight.dtype

    soft_tokens = soft_tokens.to(model_dtype)

    batch_inputs = []

    for i in range(soft_tokens.size(0)):
        soft_i = soft_tokens[i]  # [S, D]

        # -------------------------
        # CASE 1 — RAG available
        # -------------------------
        if retrieved_texts[i] is not None:
            context = "\n".join(retrieved_texts[i])

            prompt = (
                "Context:\n"
                f"{context}\n\n"
                "Task: Describe the molecule."
            )

        # -------------------------
        # CASE 2 — NO RAG
        # -------------------------
        else:
            prompt = "Describe the molecule."

        tok = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            padding=False,
        ).to(device)

        emb_prompt = emb_layer(tok.input_ids).squeeze(0)
        emb_prompt = emb_prompt.to(model_dtype)

        fused = torch.cat(
            [soft_i, emb_prompt],
            dim=0,
        )

        batch_inputs.append(fused)

    return torch.nn.utils.rnn.pad_sequence(
        batch_inputs,
        batch_first=True,
    )
