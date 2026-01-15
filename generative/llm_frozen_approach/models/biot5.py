import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


def load_biot5(
    device: str,
    model_name: str = "QizhiPei/biot5-base",
):
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
    ).to(device)

    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    return model, tokenizer


def get_llm_dim(llm) -> int:
    return int(llm.config.d_model)
