import torch
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from data_utils import PreprocessedGraphDataset
from llm_frozen_approach.models.encoder import GraphEncoder, GraphEncoderConfig
from llm_frozen_approach.models.mapper import LinearMapper
from llm_frozen_approach.models.gpt2 import load_gpt2
from torch_geometric.data import Batch
import argparse
from tqdm import tqdm
import os 
from transformers import get_linear_schedule_with_warmup
from dataset.dataset import GraphTextDataset, GraphOnlyDataset
from torch.utils.data import DataLoader

import pandas as pd


def load_trained_model(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device)

    graph_encoder = GraphEncoder(
        hidden_dim=ckpt["hidden_dim"],
        num_layers=4,
        num_heads=4,
    ).to(device)

    mapper = LinearMapper(
        dim_graph=ckpt["hidden_dim"],
        dim_llm=ckpt["llm_dim"],
        num_soft_tokens=ckpt["num_soft_tokens"]
    )


    graph_encoder.load_state_dict(ckpt["graph_encoder_state_dict"])
    mapper.load_state_dict(ckpt["mapper_state_dict"])

    graph_encoder.eval()
    mapper.eval()

    return graph_encoder, mapper



@torch.no_grad()
def generate_from_graphs(
    graphs,
    graph_encoder,
    mapper,
    llm,
    tokenizer,
    device,
    max_new_tokens=128,
    num_beams=4,
):
    graph_batch = Batch.from_data_list(graphs).to(device)

    # 1. Encode graph
    graph_emb = graph_encoder(graph_batch)          # (B, Dg)
    soft_prompt = mapper(graph_emb)                 # (B, S, Dllm)

    B, S, _ = soft_prompt.shape

    # 2. Attention mask for soft tokens
    attention_mask = torch.ones(
        (B, S),
        device=device,
        dtype=torch.long,
    )

    # 3. Generation
    end_id = tokenizer.convert_tokens_to_ids("<END>")

    outputs = llm.generate(
        inputs_embeds=soft_prompt,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        num_beams=num_beams,
        do_sample=False,
        eos_token_id=[tokenizer.eos_token_id, end_id],
        pad_token_id=tokenizer.pad_token_id,
    )


    texts = tokenizer.batch_decode(
        outputs,
        skip_special_tokens=True
    )

    return texts





def run_inference_on_test(
    test_graphs_path,
    checkpoint_path,
    device,
    batch_size=8,
    max_new_tokens=128,
):
    test_graphs = PreprocessedGraphDataset(test_graphs_path)

    test_loader = DataLoader(
        test_graphs,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda x: x,
    )

    llm, tokenizer = load_gpt2(device)
    graph_encoder, mapper = load_trained_model(checkpoint_path, device)

    graph_encoder.to(device)
    mapper.to(device)
    llm.to(device)

    graph_encoder.eval()
    mapper.eval()
    llm.eval()

    results = []

    for graphs in tqdm(test_loader, desc="Inference"):
        texts = generate_from_graphs(
            graphs=graphs,
            graph_encoder=graph_encoder,
            mapper=mapper,
            llm=llm,
            tokenizer=tokenizer,
            device=device,
            max_new_tokens=max_new_tokens,
        )

        for g, t in zip(graphs, texts):
            results.append(
                {
                    "ID": g.id,
                    "description": t,
                }
            )

    return results



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test_graphs_path",
        type=str,
        required=True,
        help="Path to test graphs .pkl file",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to trained model checkpoint",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Computation device",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for inference",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=128,
        help="Maximum number of new tokens to generate",
    )

    args = parser.parse_args()

    results = run_inference_on_test(
        test_graphs_path=args.test_graphs_path,
        checkpoint_path=args.checkpoint_path,
        device=args.device,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
    )

    

    df = pd.DataFrame(results)
    df.to_csv("submission.csv", index=False)
    print("submission.csv saved")

    