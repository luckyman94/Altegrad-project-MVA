import argparse
import os
from collections import defaultdict

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data_utils import (
    load_id2emb,
    load_descriptions_from_graphs,
    PreprocessedGraphDataset,
    collate_fn,
)

from gine_encoder.encoder import MolGNN
from gps_encoder.encoder import GraphEncoder, GraphEncoderConfig



@torch.no_grad()
def retrieve_descriptions_topk_weighted(
    model,
    train_graphs,
    test_graphs,
    train_emb_dict,
    device,
    output_csv,
    topk=5,
    batch_size=64,
):
    train_id2desc = load_descriptions_from_graphs(train_graphs)

    train_ids = list(train_emb_dict.keys())
    train_embs = torch.stack(
        [train_emb_dict[k] for k in train_ids]
    ).to(device)
    train_embs = F.normalize(train_embs, dim=-1)

    print(f"Train set size: {len(train_ids)}")

    test_ds = PreprocessedGraphDataset(test_graphs)
    test_dl = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    test_embs, test_ids = [], []
    ptr = 0

    for graphs in test_dl:
        graphs = graphs.to(device)
        z = model(graphs)
        z = F.normalize(z, dim=-1)

        test_embs.append(z)

        bs = graphs.num_graphs
        test_ids.extend(test_ds.ids[ptr : ptr + bs])
        ptr += bs

    test_embs = torch.cat(test_embs, dim=0)
    print(f"Encoded {test_embs.size(0)} test molecules")

    sims = test_embs @ train_embs.t()
    topk_vals, topk_idx = sims.topk(topk, dim=-1)

    topk_vals = topk_vals.cpu()
    topk_idx = topk_idx.cpu()

    results = []

    for i, test_id in enumerate(test_ids):
        scores = defaultdict(float)

        for r in range(topk):
            train_id = train_ids[topk_idx[i, r]]
            sim = topk_vals[i, r].item()
            scores[train_id] += sim  

        best_train_id = max(scores, key=scores.get)
        final_desc = train_id2desc[best_train_id]

        results.append({
            "ID": test_id,
            "description": final_desc,
        })

        if i < 3:
            print(f"\nTest ID {test_id}")
            for tid, sc in sorted(scores.items(), key=lambda x: -x[1]):
                print(f"  {tid}: {sc:.3f}")
            print("→ Selected:")
            print(final_desc[:120], "...")

    
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"\nSaved submission CSV → {output_csv}")

    return df




def parse_args():
    p = argparse.ArgumentParser("Graph-to-text top-k weighted retrieval")

    p.add_argument("--encoder", type=str, required=True,
                   choices=["gine", "gps"],
                   help="Graph encoder type")

    p.add_argument("--model_path", type=str, required=True)
    p.add_argument("--train_graphs", type=str, required=True)
    p.add_argument("--test_graphs", type=str, required=True)
    p.add_argument("--train_emb", type=str, required=True)

    p.add_argument("--topk", type=int, default=5,
                   help="Number of neighbors used in weighted vote")

    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--output_csv", type=str, required=True)
    p.add_argument("--device", type=str, default="cuda")

    return p.parse_args()


    

def main():
    args = parse_args()

    device = torch.device(
        args.device if torch.cuda.is_available() else "cpu"
    )
    print(f"Device: {device}")

    if not os.path.exists(args.model_path):
        raise FileNotFoundError(args.model_path)

    
    train_emb = load_id2emb(args.train_emb)
    emb_dim = len(next(iter(train_emb.values())))

    
    print(f"Loading model from {args.model_path}")

    if args.encoder == "gine":
        model = MolGNN(out_dim=emb_dim).to(device)
        model.load_state_dict(
            torch.load(args.model_path, map_location=device)
        )
        model.eval()

    elif args.encoder == "gps":
        cfg = GraphEncoderConfig(out_dim=emb_dim)
        model = GraphEncoder(cfg).to(device)
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        model.eval()

    else:
        raise ValueError(f"Unknown encoder {args.encoder}")

    
    retrieve_descriptions_topk_weighted(
        model=model,
        train_graphs=args.train_graphs,
        test_graphs=args.test_graphs,
        train_emb_dict=train_emb,
        device=device,
        output_csv=args.output_csv,
        topk=args.topk,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
