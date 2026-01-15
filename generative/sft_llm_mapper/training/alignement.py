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


def mean_pool(x):
    return x.mean(dim=1)


def train_stage1(
    graph_encoder_ckpt,
    device="cuda",
    epochs=5,
    lr=3e-4,
):
    # -----------------------
    # Load frozen components
    # -----------------------
    graph_encoder = GraphEncoder.load_from_checkpoint(graph_encoder_ckpt)
    graph_encoder.to(device).eval()
    for p in graph_encoder.parameters():
        p.requires_grad = False

    llm, tokenizer = load_gpt2(device, use_lora=False)
    llm.eval()
    for p in llm.parameters():
        p.requires_grad = False

    mapper = LinearMapper(
        dim_graph=graph_encoder.cfg.out_dim,
        dim_llm=llm.config.n_embd,
        num_soft_tokens=4,
    ).to(device)

    optimizer = torch.optim.AdamW(mapper.parameters(), lr=lr)

    # -----------------------
    # Data
    # -----------------------
    dataset = GraphTextDataset(...)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # -----------------------
    # Training loop
    # -----------------------
    for epoch in range(epochs):
        total_loss = 0.0

        for batch in tqdm(loader):
            graphs = batch["graph"]
            input_ids = batch["input_ids"].to(device)

            graph_batch = Batch.from_data_list(graphs).to(device)

            with torch.no_grad():
                graph_emb = graph_encoder(graph_batch)
                text_emb = llm.get_input_embeddings()(input_ids)
                text_emb = text_emb.mean(dim=1)

            soft_prompt = mapper(graph_emb)
            soft_mean = mean_pool(soft_prompt)

            loss = F.mse_loss(soft_mean, text_emb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"[Stage1] Epoch {epoch} | loss={total_loss/len(loader):.4f}")

    torch.save(
        {"mapper_state_dict": mapper.state_dict()},
        "stage1_mapper.pt"
    )
