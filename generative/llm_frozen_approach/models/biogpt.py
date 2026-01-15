import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType


def load_biogpt(
    device: str,
    model_name: str = "microsoft/biogpt",
):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    llm = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
    ).to(device)

    
    
    llm.eval()
    for p in llm.parameters():
        p.requires_grad = False

    return llm, tokenizer


def get_llm_dim(llm) -> int:
    return int(getattr(llm.config, "hidden_size"))
