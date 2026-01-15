import torch
from tqdm import tqdm
from pathlib import Path

import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))
from transformers.modeling_outputs import BaseModelOutput
from generative_modeling_via_latent_space.dataset.load_data import load_graphs
from generative_modeling_via_latent_space.models.latent_prefix_ae import LatentPrefixAE
from generative_modeling_via_latent_space.dataset import GraphLatentDataset

from generative_modeling_via_latent_space.constants import *
from generative_modeling_via_latent_space.losses import latent_loss
from huggingface_hub import hf_hub_download
from torch_geometric.loader import DataLoader as PyGDataLoader
from generative_modeling_via_latent_space.models.graph_to_latent import GraphToLatent
from generative_modeling_via_latent_space.metrics.eval_bleu_rouge import evaluate_bleu_rouge

def train_epoch(model, loader, optimizer, latent_ae, lambda_dec):
    model.train()
    total = 0.0

    for graph, latent, text in tqdm(loader, leave=False):
        graph, latent = graph.to(DEVICE), latent.to(DEVICE)

        pred_latent = model(graph)
        pred_latent += 0.02 * torch.randn_like(pred_latent)

        loss_lat = latent_loss(pred_latent, latent)

        tok = latent_ae.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=TRAIN_MAX_LEN,
            return_tensors="pt"
        ).to(DEVICE)

        dec_out = latent_ae.model(
            encoder_outputs=BaseModelOutput(last_hidden_state=pred_latent),
            labels=tok.input_ids
        )

        loss = loss_lat + lambda_dec * dec_out.loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total += loss.item()

    return total / len(loader)


@torch.no_grad()
def eval_epoch(model, loader, latent_ae, lambda_dec):
    model.eval()
    total = 0.0

    for graph, latent, text in loader:
        graph, latent = graph.to(DEVICE), latent.to(DEVICE)

        pred_latent = model(graph)
        loss_lat = latent_loss(pred_latent, latent)

        tok = latent_ae.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=TRAIN_MAX_LEN,
            return_tensors="pt"
        ).to(DEVICE)

        dec_out = latent_ae.model(
            encoder_outputs=BaseModelOutput(last_hidden_state=pred_latent),
            labels=tok.input_ids
        )

        total += (loss_lat + lambda_dec * dec_out.loss).item()

    return total / len(loader)

def unfreeze_last_decoder_layers(model, n=2):
    for p in model.model.decoder.parameters():
        p.requires_grad = False
    for block in model.model.decoder.block[-n:]:
        for p in block.parameters():
            p.requires_grad = True



def lambda_schedule(epoch, max_epoch):
    return 1.0 - 0.7 * (epoch / max_epoch)


def main():
    
    train_graphs, val_graphs, train_id2text, val_id2text = load_graphs()

    
    latent_ae = LatentPrefixAE()

    ckpt = hf_hub_download(
    repo_id="TheoSG/Altegrad",
    filename="LatentPrefixAE.pt",
    repo_type="dataset"
)
    state = torch.load(ckpt, map_location=DEVICE)
    latent_ae.load_state_dict(state["model_state_dict"], strict=False)
    latent_ae.eval()

    print("LatentPrefixAE loaded correctly (0.99 BLEU model)")

    train_ds = GraphLatentDataset(train_graphs, train_id2text, latent_ae)
    val_ds = GraphLatentDataset(val_graphs, val_id2text, latent_ae)

    train_loader = PyGDataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = PyGDataLoader(val_ds, batch_size=BATCH_SIZE)

    model = GraphToLatent().to(DEVICE)

    unfreeze_last_decoder_layers(latent_ae, n=2)

    optimizer = torch.optim.AdamW(
        [
            {"params": model.parameters(), "lr": 2e-4},
            {
                "params": filter(lambda p: p.requires_grad, latent_ae.parameters()),
                "lr": 1e-5
            }
        ]
    )

    path = hf_hub_download(
    repo_id="TheoSG/Altegrad",
    filename="graph2latent_epoch5 (1).pt",
    repo_type="dataset"
    )

    EPOCHS = 10

    for epoch in range(5, EPOCHS + 1):
        lambda_dec = lambda_schedule(epoch, EPOCHS)

        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            latent_ae,
            lambda_dec
        )

        val_loss = eval_epoch(
            model,
            val_loader,
            latent_ae,
            lambda_dec
        )

        torch.save(
            {"graph2latent": model.state_dict(), "epoch": epoch},
            f"{SAVE_DIR}/graph2latent_epoch{epoch}.pt"
        )

        print(
            f"\nEpoch {epoch:02d} | "
            f"λ={lambda_dec:.3f} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f}"
        )
    
    bleu, rouge = evaluate_bleu_rouge(
    model,
    val_loader,
    latent_ae,
    max_print=10
    )

    print(f"\nBLEU = {bleu:.4f}")
    print(f"ROUGE = {rouge:.4f}")


if __name__ == "__main__":
    main()