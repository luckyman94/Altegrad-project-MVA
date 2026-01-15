import argparse
import sys
from pathlib import Path

import torch
import numpy as np
import faiss
from tqdm import tqdm
from torch_geometric.data import Batch

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from data_utils import PreprocessedGraphDataset
from sft_llm_mapper.models.encoder import GraphEncoder, GraphEncoderConfig


def load_graph_encoder_from_ckpt(ckpt_path: str, device: str):
    ckpt = torch.load(ckpt_path, map_location=device)

    cfg = GraphEncoderConfig(
        hidden_dim=ckpt["gnn_hidden_dim"],
        out_dim=ckpt["gnn_out_dim"],
        num_layers=ckpt["num_layers"],
        num_heads=ckpt["num_heads"],
        dropout=ckpt["dropout"],
        attn_type=ckpt["attn_type"],
        pool=ckpt["pool"],
        normalize_out=ckpt["normalize_out"],
    )

    model = GraphEncoder(cfg).to(device)
    model.load_state_dict(ckpt["graph_encoder_state_dict"])
    model.eval()

    print(f"Loaded graph encoder from {ckpt_path}")

    return model, cfg.out_dim


@torch.no_grad()
def build_faiss_index(
    graph_encoder,
    train_ds,
    device,
    out_index_path: str,
):
    graph_encoder.eval().to(device)

    embeddings = []
    texts = []

    for g in tqdm(train_ds, desc="Encoding train graphs"):
        batch = Batch.from_data_list([g]).to(device)
        z = graph_encoder(batch)
        z = z.cpu().numpy()
        z = z / np.linalg.norm(z, axis=1, keepdims=True) 

        embeddings.append(z)
        texts.append(g.description)

    X = np.vstack(embeddings).astype("float32")
    dim = X.shape[1]

    print(f"\n📐 Building FAISS index (N={X.shape[0]}, dim={dim})")

    index = faiss.IndexFlatIP(dim)  
    index.add(X)

    faiss.write_index(index, out_index_path)
    torch.save(texts, out_index_path + ".texts.pt")

    print(f"FAISS index saved to: {out_index_path}")
    print(f"Text corpus saved to: {out_index_path}.texts.pt")


def main():
    p = argparse.ArgumentParser("Build FAISS index over graph embeddings")

    p.add_argument("--train_data", required=True, help="Path to train_graphs.pkl")
    p.add_argument("--encoder_ckpt", required=True, help="Path to graph encoder checkpoint")
    p.add_argument("--out_index", default="graph_faiss.index", help="Output FAISS index path")
    p.add_argument("--device", default="cuda")

    args = p.parse_args()
    device = args.device if torch.cuda.is_available() else "cpu"

    print(f"Building FAISS graph index on device: {device}\n")

    train_ds = PreprocessedGraphDataset(args.train_data)
    print(f"Loaded {len(train_ds)} training graphs")

    graph_encoder, _ = load_graph_encoder_from_ckpt(
        args.encoder_ckpt,
        device,
    )

    build_faiss_index(
        graph_encoder=graph_encoder,
        train_ds=train_ds,
        device=device,
        out_index_path=args.out_index,
    )


if __name__ == "__main__":
    main()
