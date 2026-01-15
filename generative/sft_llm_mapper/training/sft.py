import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch_geometric.data import Batch
from tqdm import tqdm

from sft_llm_mapper.models.encoder import GraphEncoder, GraphEncoderConfig
from sft_llm_mapper.models.mapper import LinearMapper
from sft_llm_mapper.models.gpt2 import load_gpt2  
from data_utils import PreprocessedGraphDataset
from sft_llm_mapper.dataset.dataset import GraphTextDataset

import torch
from torch_geometric.data import Batch
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler

def train_stage2(
    graph_encoder_ckpt,
    mapper_ckpt,
    device="cuda",
    epochs=10,
):
    # -----------------------
    # Load frozen GraphEncoder
    # -----------------------
    graph_encoder = GraphEncoder.load_from_checkpoint(graph_encoder_ckpt)
    graph_encoder.eval()
    for p in graph_encoder.parameters():
        p.requires_grad = False

    # -----------------------
    # Load LLM with LoRA
    # -----------------------
    llm, tokenizer = load_gpt2(
        device,
        use_lora=True,
    )

    mapper = LinearMapper(...)
    mapper.load_state_dict(torch.load(mapper_ckpt)["mapper_state_dict"])

    optimizer = torch.optim.AdamW(
        [
            {"params": mapper.parameters(), "lr": 3e-4},
            {"params": llm.parameters(), "lr": 1e-5},
        ]
    )

    scaler = GradScaler()

    for epoch in range(epochs):
        for batch in tqdm(loader):
            graphs = batch["graph"]
            input_ids = batch["input_ids"].to(device)

            graph_batch = Batch.from_data_list(graphs).to(device)
            graph_emb = graph_encoder(graph_batch)
            soft_prompt = mapper(graph_emb)

            text_emb = llm.get_input_embeddings()(input_ids)

            inputs_embeds = torch.cat(
                [soft_prompt, text_emb],
                dim=1,
            )

            labels = torch.cat(
                [
                    torch.full((input_ids.size(0), soft_prompt.size(1)), -100).to(device),
                    input_ids,
                ],
                dim=1,
            )

            with autocast(dtype=torch.bfloat16):
                outputs = llm(
                    inputs_embeds=inputs_embeds,
                    labels=labels,
                )
                loss = outputs.loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        print(f"[Stage2] Epoch {epoch} | loss={loss.item():.4f}")
