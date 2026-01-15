import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType


def load_biogpt(
    device: str,
    model_name: str = "microsoft/biogpt",
    use_lora: bool = False,
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
):
    

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    llm = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype= torch.float32
    ).to(device)

    for p in llm.parameters():
        p.requires_grad = False

    if use_lora:
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "out_proj",
                "fc1",
                "fc2",
            ],
        )

        llm = get_peft_model(llm, lora_config)

        try:
            llm.print_trainable_parameters()
        except Exception:
            pass

        llm.train()
    else:
        llm.eval()

    return llm, tokenizer


def get_llm_dim(llm) -> int:
    return int(llm.config.hidden_size)
