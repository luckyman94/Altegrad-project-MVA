from .gpt2 import load_gpt2
from .biogpt import load_biogpt

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel


class FrozenLLMEmbedder(nn.Module):
    """
    Text -> embedding wrapper
    Compatible GPT-2 / BioGPT
    Used ONLY in Stage-1 alignment
    """

    def __init__(self, model_name: str, device="cuda"):
        super().__init__()
        self.device = device

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.model.eval()

        for p in self.model.parameters():
            p.requires_grad = False

        self.hidden_size = self.model.config.hidden_size

    @torch.no_grad()
    def encode(self, texts):
        toks = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)

        h = self.model(**toks).last_hidden_state   # [B, L, d]
        emb = h.mean(dim=1)                        # [B, d]
        emb = emb / (emb.norm(dim=-1, keepdim=True) + 1e-12)
        return emb


def load_llm(llm_name: str, device: str, use_lora: bool = False):
    llm_name = llm_name.lower()

    if llm_name == "gpt2":
        llm, tokenizer = load_gpt2(
            device=device,
            use_lora=use_lora,
        )
        llm_dim = llm.config.n_embd

    elif llm_name == "biogpt":
        llm, tokenizer = load_biogpt(
            device=device,
            use_lora=use_lora,
        )
        llm_dim = llm.config.hidden_size

    else:
        raise ValueError(
            f"Unknown LLM '{llm_name}'. Available: gpt2 | biogpt"
        )

    return llm, tokenizer, llm_dim


def load_llm_embedder(llm_name: str, device: str):
    """
    For Stage-1 only (alignment)
    """
    if llm_name == "gpt2":
        return FrozenLLMEmbedder("gpt2", device)
    elif llm_name == "biogpt":
        return FrozenLLMEmbedder("microsoft/biogpt", device)
    else:
        raise ValueError(f"Unknown LLM '{llm_name}'")

