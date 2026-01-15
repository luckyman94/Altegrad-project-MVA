import argparse
import os
from collections import defaultdict
import torch
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
from losses.infonce import infonce_loss

from data_utils import (
    load_id2emb,
    load_descriptions_from_graphs,
    PreprocessedGraphDataset,
    collate_fn,
)

from gine_encoder.encoder import MolGNN
from gps_encoder.encoder import load_graph_encoder_from_checkpoint



BLOCKS = ["intro", "structure", "role", "taxonomy"]

DEFAULT_BLOCK_WEIGHTS = {
    "intro": 1.0,
    "structure": 0.7,
    "role": 1.2,
    "taxonomy": 0.5,
}


@torch.no_grad()
def retrieve_multiblock(
    model,
    train_graphs,
    test_graphs,
    train_embs_by_block,
    device,
    output_csv,
    top_k=5,
    block_weights=None,
    batch_size=64,
):
    if block_weights is None:
        block_weights = DEFAULT_BLOCK_WEIGHTS

    train_id2desc = load_descriptions_from_graphs(train_graphs)
    train_ids = list(next(iter(train_embs_by_block.values())).keys())

    train_emb_tensors = {
        block: torch.stack([train_embs_by_block[block][i] for i in train_ids]).to(device)
        for block in BLOCKS
    }

    test_ds = PreprocessedGraphDataset(test_graphs)
    test_dl = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )

    test_embs = []
    test_ids = []
    ptr = 0

    for graphs in tqdm(test_dl, desc="Encoding test graphs"):
        graphs = graphs.to(device)
        z = model(graphs)
        z = F.normalize(z, dim=-1)
        test_embs.append(z)

        bs = graphs.num_graphs
        test_ids.extend(test_ds.ids[ptr : ptr + bs])
        ptr += bs

    test_embs = torch.cat(test_embs, dim=0)

    results = []

    for i, test_id in tqdm(
        enumerate(test_ids),
        total=len(test_ids),
        desc="Retrieving (multi-block)",
    ):
        z = test_embs[i]
        scores = defaultdict(float)

        for block in BLOCKS:
            sims = z @ train_emb_tensors[block].T
            topk_scores, topk_idx = sims.topk(top_k)

            for idx, sim in zip(topk_idx.tolist(), topk_scores.tolist()):
                train_id = train_ids[idx]
                scores[train_id] += block_weights[block] * sim

        best_train_id = max(scores, key=scores.get)
        final_desc = train_id2desc[best_train_id]

        results.append({
            "ID": test_id,
            "description": final_desc,
        })

        if i < 3:
            print(f"\nTest {test_id}")
            for k, v in sorted(scores.items(), key=lambda x: -x[1])[:5]:
                print(f"  {k}: {v:.3f}")
            print("→ Selected:")
            print(final_desc[:120], "...")

    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"\nSaved submission to {output_csv}")

    return df



def parse_args():
    p = argparse.ArgumentParser("Graph-to-text multi-block retrieval")

    p.add_argument("--encoder", type=str, choices=["gine", "gps"], required=True)
    p.add_argument("--model_path", type=str, required=True)

    p.add_argument("--train_graphs", type=str, required=True)
    p.add_argument("--test_graphs", type=str, required=True)

    p.add_argument("--train_intro_emb", type=str, required=True)
    p.add_argument("--train_structure_emb", type=str, required=True)
    p.add_argument("--train_role_emb", type=str, required=True)
    p.add_argument("--train_taxonomy_emb", type=str, required=True)

    p.add_argument("--output_csv", type=str, required=True)
    p.add_argument("--top_k", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--device", type=str, default="cuda")

    return p.parse_args()



def main():
    args = parse_args()

    device = torch.device(
        args.device if torch.cuda.is_available() else "cpu"
    )
    print(f"Device: {device}")

    train_embs_by_block = {
        "intro": load_id2emb(args.train_intro_emb),
        "structure": load_id2emb(args.train_structure_emb),
        "role": load_id2emb(args.train_role_emb),
        "taxonomy": load_id2emb(args.train_taxonomy_emb),
    }

    emb_dim = len(next(iter(train_embs_by_block["intro"].values())))

    print(f"Loading model from {args.model_path}")

    if args.encoder == "gine":
        model = MolGNN(out_dim=emb_dim).to(device)
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        model.eval()

    elif args.encoder == "gps":
        model = load_graph_encoder_from_checkpoint(
            model_path=args.model_path,
            device=device,
        )

    else:
        raise ValueError(f"Unknown encoder {args.encoder}")

    retrieve_multiblock(
        model=model,
        train_graphs=args.train_graphs,
        test_graphs=args.test_graphs,
        train_embs_by_block=train_embs_by_block,
        device=device,
        output_csv=args.output_csv,
        top_k=args.top_k,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
