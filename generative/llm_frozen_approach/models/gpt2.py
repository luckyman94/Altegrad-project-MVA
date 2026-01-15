import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer


from peft import LoraConfig, get_peft_model

def load_gpt2(device="cuda", model_name="gpt2", use_lora=True):
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    model = GPT2LMHeadModel.from_pretrained(model_name)
    model.to(device)

    if use_lora:
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=[
                "c_attn",  
                "c_proj",  
                "c_fc",    
            ],
        )

        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    model.train()   

    return model, tokenizer
