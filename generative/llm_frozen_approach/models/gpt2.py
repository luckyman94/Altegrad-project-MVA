import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

def load_gpt2(device="cuda", model_name="gpt2"):
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    model = GPT2LMHeadModel.from_pretrained(model_name).to(device)

    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    return model, tokenizer
