from gpt2 import load_gpt2
from biogpt import load_biogpt


def load_llm(llm_name: str, device: str):
    llm_name = llm_name.lower()

    if llm_name == "gpt2":
        llm, tokenizer = load_gpt2(device)
        llm_dim = llm.config.n_embd
        return llm, tokenizer, llm_dim

    elif llm_name == "biogpt":
        llm, tokenizer = load_biogpt(device)
        llm_dim = llm.config.hidden_size
        return llm, tokenizer, llm_dim


    else:
        raise ValueError(
            f"Unknown LLM '{llm_name}'. "
            f"Available: gpt2 | biogpt "
        )
