import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from peft import LoraConfig, get_peft_model, TaskType


def load_gpt2(
    device="cuda",
    model_name="gpt2",
    lora_r=8,
    lora_alpha=16,
    lora_dropout=0.05,
):
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    model = GPT2LMHeadModel.from_pretrained(model_name).to(device)

    
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        target_modules=[
            "c_attn",  
            "c_proj",  
            "c_fc",    
        ],
    )

    model = get_peft_model(model, lora_config)

    try:
        model.print_trainable_parameters()
    except Exception:
        pass

    model.train()

    return model, tokenizer
