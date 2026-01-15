import torch
from pathlib import Path
import sys
from torch_geometric.data import Batch
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
import argparse

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from data_utils import PreprocessedGraphDataset
from llm_frozen_approach.models.encoder import GraphEncoder, GraphEncoderConfig
from llm_frozen_approach.models.mapper import LinearMapper
from llm_frozen_approach.models.biot5 import load_biot5


def load_trained_model(checkpoint_path, device, llm_dim):
    ckpt = torch.load(checkpoint_path, map_location=device)

    cfg = GraphEncoderConfig(
        hidden_dim=ckpt["gnn_hidden_dim"],
        out_dim=ckpt["gnn_out_dim"],
        num_layers=4,
        num_heads=4,
        dropout=0.1,
        pool="mean",
        normalize_out=True,
    )

    graph_encoder = GraphEncoder(cfg).to(device)

    mapper = LinearMapper(
        dim_graph=ckpt["gnn_out_dim"],
        dim_llm=llm_dim,
        num_soft_tokens=ckpt["num_soft_tokens"],
    ).to(device)

    graph_encoder.load_state_dict(ckpt["graph_encoder"])
    mapper.load_state_dict(ckpt["mapper"])

    graph_encoder.eval()
    mapper.eval()

    return graph_encoder, mapper


@torch.no_grad()
def generate_from_graphs_biot5(
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

    graph_emb = graph_encoder(graph_batch)   
    soft_prompt = mapper(graph_emb)          

    B, S, D = soft_prompt.shape

    prompt_text = "Describe the following molecule:"
    prompt_ids = tokenizer(
        prompt_text,
        return_tensors="pt",
        add_special_tokens=False,
    ).input_ids.to(device)

    prompt_embeds = llm.encoder.embed_tokens(prompt_ids)
    prompt_embeds = prompt_embeds.repeat(B, 1, 1)

    encoder_inputs_embeds = torch.cat(
        [soft_prompt, prompt_embeds],
        dim=1,
    )

    encoder_attention_mask = torch.ones(
        encoder_inputs_embeds.size()[:-1],
        device=device,
        dtype=torch.long,
    )

    encoder_outputs = llm.get_encoder()(
        inputs_embeds=encoder_inputs_embeds,
        attention_mask=encoder_attention_mask,
        return_dict=True,
    )

    outputs = llm.generate(
        encoder_outputs=encoder_outputs,
        attention_mask=encoder_attention_mask,
        max_new_tokens=max_new_tokens,
        num_beams=num_beams,
        do_sample=False,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )

    texts = tokenizer.batch_decode(
        outputs,
        skip_special_tokens=True,
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

    llm, tokenizer = load_biot5(device)
    llm_dim = llm.config.d_model
    graph_encoder, mapper = load_trained_model(checkpoint_path, device, llm_dim)

    llm.eval()
    graph_encoder.eval()
    mapper.eval()

    results = []

    for graphs in tqdm(test_loader, desc="BioT5 inference"):
        texts = generate_from_graphs_biot5(
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
    parser = argparse.ArgumentParser("BioT5 inference")

    parser.add_argument("--test_graphs_path", type=str, required=True)
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=128)

    args = parser.parse_args()

    results = run_inference_on_test(
        test_graphs_path=args.test_graphs_path,
        checkpoint_path=args.checkpoint_path,
        device=args.device,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
    )

    df = pd.DataFrame(results)
    df.to_csv("submission_biot5.csv", index=False)
    print("submission_biot5.csv saved")
