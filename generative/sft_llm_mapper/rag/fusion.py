import torch
from typing import List


from typing import List, Optional


@torch.no_grad()
def fuse_soft_tokens_and_rag(
    llm,
    tokenizer,
    soft_tokens: torch.Tensor,                 
    retrieved_texts: List[Optional[List[str]]],
    device: torch.device,
):

    emb_layer = llm.get_input_embeddings()
    model_dtype = emb_layer.weight.dtype

    soft_tokens = soft_tokens.to(device=device, dtype=model_dtype)

    batch_inputs = []
    batch_attn = []

    bos_id = tokenizer.bos_token_id
    if bos_id is None:
        bos_id = tokenizer.eos_token_id

    bos_emb = emb_layer(
        torch.tensor([[bos_id]], device=device)
    ).squeeze(0)  # [1, D]

    for i in range(soft_tokens.size(0)):
        soft_i = soft_tokens[i]  # [S, D]

        texts = retrieved_texts[i]

        has_context = (
            texts is not None
            and len(texts) > 0
            and any(t.strip() != "" for t in texts)
        )

        if has_context:
            context = "\n".join(t.strip() for t in texts if t.strip() != "")
            prompt = (
                "Context:\n"
                f"{context}\n\n"
                "Task: Describe the molecule using the context above.\n"
            )
        else:
            prompt = (
                "Task: Describe the molecule based on its molecular graph.\n"
                "Focus on functional groups and chemical properties.\n"
            )

        tok = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            padding=False,
        ).to(device)

        emb_prompt = emb_layer(tok.input_ids).squeeze(0).to(model_dtype)

        fused = torch.cat(
            [soft_i, emb_prompt, bos_emb],
            dim=0,
        )  

        batch_inputs.append(fused)
        batch_attn.append(torch.ones(fused.size(0), device=device))

    inputs_embeds = torch.nn.utils.rnn.pad_sequence(
        batch_inputs,
        batch_first=True,
    )

    attention_mask = torch.nn.utils.rnn.pad_sequence(
        batch_attn,
        batch_first=True,
    )

    return inputs_embeds, attention_mask
