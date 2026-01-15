import argparse
import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(0, str(ROOT))
from losses.infonce import infonce_loss
from data_utils import PreprocessedGraphDataset, collate_fn, load_id2emb
from encoder import MolGNN

def parse_args():
    parser = argparse.ArgumentParser("Graph–Text InfoNCE Training")

    parser.add_argument("--train_graphs", type=str, required=True,
                        help="Path to preprocessed train graphs (.pkl)")
    parser.add_argument("--val_graphs", type=str, default=None,
                        help="Path to preprocessed val graphs (.pkl)")
    parser.add_argument("--train_emb", type=str, required=True,
                        help="CSV mapping train id -> text embedding")
    parser.add_argument("--val_emb", type=str, default=None,
                        help="CSV mapping val id -> text embedding")
    parser.add_argument("--output_model", type=str, default="model_checkpoint.pt",
                        help="Where to save trained model")

    
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--temperature", type=float, default=0.07)

    
    parser.add_argument("--device", type=str, default="cuda")

    return parser.parse_args()

def train_epoch_InfoNCE(mol_enc, loader, optimizer, device):
    mol_enc.train()
    total_loss, total = 0.0, 0

    temperature = 0.07

    for graphs, text_emb in loader:
        graphs = graphs.to(device)
        text_emb = text_emb.to(device)

        mol_vec = mol_enc(graphs)
        txt_vec = F.normalize(text_emb, dim=-1)

        loss = infonce_loss(mol_vec, txt_vec, temperature=temperature)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_size = graphs.num_graphs
        total_loss += loss.item() * batch_size
        total += batch_size

    return total_loss / total



@torch.no_grad()
def eval_retrieval(data_path, emb_dict, mol_enc, device, temperature=0.07):
    ds = PreprocessedGraphDataset(data_path, emb_dict)
    dl = DataLoader(ds, batch_size=64, shuffle=False, collate_fn=collate_fn)

    all_mol, all_txt = [], []

    for graphs, text_emb in dl:
        graphs = graphs.to(device)
        text_emb = text_emb.to(device)

        mol_vec = mol_enc(graphs)
        txt_vec = F.normalize(text_emb, dim=-1)

        all_mol.append(mol_vec)
        all_txt.append(txt_vec)

    all_mol = torch.cat(all_mol, dim=0)
    all_txt = torch.cat(all_txt, dim=0)

    sims = (all_txt @ all_mol.t()) / temperature

    ranks = sims.argsort(dim=-1, descending=True)

    N = sims.size(0)
    device = sims.device
    correct = torch.arange(N, device=device)

    pos = (ranks == correct.unsqueeze(1)).nonzero()[:, 1] + 1

    mrr = (1.0 / pos.float()).mean().item()

    results = {"MRR": mrr}
    for k in (1, 5, 10):
        hitk = (pos <= k).float().mean().item()
        results[f"R@{k}"] = hitk
        results[f"Hit@{k}"] = hitk

    return results


def main():
    args = parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    #Load embeddings
    train_emb = load_id2emb(args.train_emb)
    val_emb = load_id2emb(args.val_emb) if args.val_emb and os.path.exists(args.val_emb) else None

    emb_dim = len(next(iter(train_emb.values())))
    print(f"Embedding dim: {emb_dim}")

    if not os.path.exists(args.train_graphs):
        print(f"Error: Preprocessed graphs not found at {args.train_graphs}")
        return

    #Dataloader
    train_ds = PreprocessedGraphDataset(args.train_graphs, train_emb)
    train_dl = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )

    #Model
    mol_enc = MolGNN(out_dim=emb_dim).to(device)
    optimizer = torch.optim.Adam(mol_enc.parameters(), lr=args.lr)

    #Training loop
    for ep in range(args.epochs):
        train_loss = train_epoch_InfoNCE(
            mol_enc,
            train_dl,
            optimizer,
            device
        )

        if args.val_graphs and val_emb and os.path.exists(args.val_graphs):
            val_scores = eval_retrieval(
                args.val_graphs,
                val_emb,
                mol_enc,
                device,
                temperature=args.temperature
            )
        else:
            val_scores = {}

        print(
            f"Epoch {ep+1}/{args.epochs} "
            f"- loss={train_loss:.4f} "
            f"- val={val_scores}"
        )

    #Save model
    torch.save(mol_enc.state_dict(), args.output_model)
    print(f"\nModel saved to {args.output_model}")



