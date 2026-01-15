import argparse
import os
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
from gps_encoder.encoder import GraphEncoder, GraphEncoderConfig, load_graph_encoder_from_checkpoint



@torch.no_grad()
def retrieve_descriptions_top_k(
    model,
    train_graphs,
    test_graphs,
    train_emb_dict,
    device,
    output_csv,
    topk=5,
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
        test_ds, batch_size=64, shuffle=False, collate_fn=collate_fn
    )

    test_embs, test_ids = [], []

    for graphs in test_dl:
        graphs = graphs.to(device)
        emb = model(graphs)
        test_embs.append(emb)

        start = len(test_ids)
        test_ids.extend(test_ds.ids[start : start + graphs.num_graphs])

    test_embs = torch.cat(test_embs, dim=0)
    print(f"Encoded {test_embs.size(0)} test molecules")

    sims = test_embs @ train_embs.t()

    topk_vals, topk_idx = sims.topk(topk, dim=-1)
    topk_idx = topk_idx.cpu()
    topk_vals = topk_vals.cpu()

    results = []
    for i, test_id in enumerate(test_ids):
        for rank in range(topk):
            train_id = train_ids[topk_idx[i, rank]]
            results.append({
                "test_id": test_id,
                "rank": rank + 1,
                "train_id": train_id,
                "score": topk_vals[i, rank].item(),
                "description": train_id2desc[train_id],
            })

        if i < 3:
            print(f"\nTest ID {test_id}")
            for r in range(min(topk, 3)):
                tid = train_ids[topk_idx[i, r]]
                print(f"  #{r+1} → {tid} | score={topk_vals[i,r]:.3f}")
                print(train_id2desc[tid][:120], "...")

    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"\nSaved top-{topk} retrievals for {len(test_ids)} molecules → {output_csv}")

    return df




def parse_args():
    p = argparse.ArgumentParser("Graph-to-text retrieval")

    p.add_argument("--encoder", type=str, required=True,
                   choices=["gine", "gps"],
                   help="Graph encoder type")

    p.add_argument("--model_path", type=str, required=True)
    p.add_argument("--train_graphs", type=str, required=True)
    p.add_argument("--test_graphs", type=str, required=True)
    p.add_argument("--train_emb", type=str, required=True)
    p.add_argument("--topk", type=int, default=1,
               help="Number of retrieved descriptions per test molecule")


    p.add_argument("--output_csv", type=str, default="retrieved_descriptions.csv")
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
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        model.eval()

    elif args.encoder == "gps":
        cfg = GraphEncoderConfig(out_dim=emb_dim)
        model = GraphEncoder(cfg).to(device)
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        model.eval()
    
    else:
        raise ValueError(f"Unknown encoder {args.encoder}")

    
    

    retrieve_descriptions_top_k(
        model=model,
        train_graphs=args.train_graphs,
        test_graphs=args.test_graphs,
        train_emb_dict=train_emb,
        device=device,
        output_csv=args.output_csv,
    )


if __name__ == "__main__":
    main()
