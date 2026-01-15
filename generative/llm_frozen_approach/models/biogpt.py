import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType


def load_biogpt(
    device: str,
    model_name: str = "microsoft/biogpt",
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    use_loRA: bool = False,
):

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    llm = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
    ).to(device)

    if not use_loRA:
        return llm, tokenizer
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        target_modules=["c_attn", "c_proj"],
    )

    llm = get_peft_model(llm, lora_cfg)

    try:
        llm.print_trainable_parameters()
    except Exception:
        pass

    return llm, tokenizer


def get_llm_dim(llm) -> int:
    return int(getattr(llm.config, "hidden_size"))
