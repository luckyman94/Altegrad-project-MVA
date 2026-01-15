import torch
from torch.utils.data import DataLoader
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
from llm_frozen_approach.models.llm_factory import load_llm



DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def save_checkpoint(graph_encoder, mapper, path, epoch, val_loss):
    os.makedirs(path, exist_ok=True)

    torch.save(
        {
            "graph_encoder_state_dict": graph_encoder.state_dict(),
            "mapper_state_dict": mapper.state_dict(),

            # architecture
            "gnn_hidden_dim": graph_encoder.cfg.hidden_dim,
            "gnn_out_dim": graph_encoder.cfg.out_dim,  
            "llm_dim": mapper.dim_llm,
            "num_soft_tokens": mapper.num_soft_tokens,

            # training
            "epoch": epoch,
            "val_loss": val_loss,
        },
        os.path.join(path, "best_graph2text.pt")
    )




def parse_args():
    parser = argparse.ArgumentParser(description="Train GraphEncoder + Mapper with GPT-2")

    parser.add_argument("--train_graphs", type=str, required=True,
                        help="Path to train_graphs.pkl")

    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--checkpoint_path", type=str, default="checkpoints",
                        help="Path to save checkpoints")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max_text_len", type=int, default=128)
    parser.add_argument("--hidden_dim", type=int, default=256)

    parser.add_argument("--device", type=str, default=None,
                        help="cuda or cpu (default: auto)")
    
    parser.add_argument(
        "--num_soft_tokens",
        type=int,
        default=4,
        help="Number of soft tokens injected into GPT-2"
    )

    parser.add_argument(
    "--llm",
    type=str,
    default="gpt2",
    choices=["gpt2", "biogpt"],
    help="Which LLM backend to use"
)


    parser.add_argument(
    "--resume",
    type=str,
    default=None,
    help="Path to checkpoint to resume training"
)


    return parser.parse_args()

def build_prompt():
    PROMPT = "Write a scientific description of the following molecule. Include: chemical class, key functional groups, biological or chemical role. End the description with <END>."



    return PROMPT
    
def main():
    args = parse_args()

    device = (
        args.device
        if args.device is not None
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )

 
    train_graphs = PreprocessedGraphDataset(args.train_graphs)
    val_graphs = PreprocessedGraphDataset(
        args.train_graphs.replace("train", "validation")
    )

    llm, tokenizer, llm_dim = load_llm(args.llm, device)

    prompt_text = build_prompt()

    prompt_ids = tokenizer(
        prompt_text,
        return_tensors="pt",
        add_special_tokens=False,
    ).input_ids.to(device)

    prompt_embeds = llm.get_input_embeddings()(prompt_ids)


    train_dataset = GraphTextDataset(
        train_graphs,
        tokenizer=tokenizer,
        max_length=args.max_text_len
    )

    val_dataset = GraphTextDataset(
        val_graphs,
        tokenizer=tokenizer,
        max_length=args.max_text_len
    )

    collate_fn = lambda x: {
        "graph": [item["graph"] for item in x],
        "input_ids": torch.stack([item["input_ids"] for item in x]),
    }

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    cfg = GraphEncoderConfig(
    hidden_dim=args.hidden_dim,
    out_dim=llm.config.n_embd,  
    num_layers=4,
    num_heads=4,
    dropout=0.1,
)

    graph_encoder = GraphEncoder(cfg).to(device)    


    mapper = LinearMapper(
        dim_graph=llm.config.n_embd,
        dim_llm=llm.config.n_embd, 
        num_soft_tokens = args.num_soft_tokens
    ).to(device)

    optimizer = torch.optim.AdamW(
    [
        {"params": graph_encoder.parameters(), "lr": 5e-5},
        {"params": mapper.parameters(), "lr": 3e-4},
    ]
    )

    start_epoch = 0

    if args.resume is not None:
        print(f"Resuming training from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)

        graph_encoder.load_state_dict(ckpt["graph_encoder_state_dict"])
        mapper.load_state_dict(ckpt["mapper_state_dict"])

        start_epoch = ckpt["epoch"]
    


    llm.eval()
    best_val_loss = float("inf")

    num_training_steps = args.epochs * len(train_loader)
    num_warmup_steps = int(0.1 * num_training_steps) 

    scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=num_warmup_steps,
    num_training_steps=num_training_steps,
)
    for epoch in range(start_epoch,args.epochs):

        graph_encoder.train()
        mapper.train()

        train_loss = 0.0

        for batch in tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{args.epochs} [train]"
        ):
            optimizer.zero_grad()

            graphs = batch["graph"]
            input_ids = batch["input_ids"].to(device)

            graph_batch = Batch.from_data_list(graphs).to(device)

            graph_emb = graph_encoder(graph_batch)
            soft_prompt = mapper(graph_emb)             

            text_emb = llm.get_input_embeddings()(input_ids)

            B = input_ids.size(0)

            prompt_embeds_batch = prompt_embeds.repeat(B, 1, 1)


            inputs_embeds = torch.cat(
                [
                    prompt_embeds_batch,  
                    soft_prompt,          
                    text_emb,             
                ],
                dim=1,
)

            labels = torch.cat(
            [
                torch.full(
                    (B, prompt_embeds_batch.size(1) + soft_prompt.size(1)),
                    -100,
                    device=device,
                ),
                input_ids,
            ],
            dim=1,
        )
            
            attention_mask = torch.ones(
            inputs_embeds.size()[:-1],
            device=device,
        )



            outputs = llm(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                labels=labels,
            )

            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        graph_encoder.eval()
        mapper.eval()

        val_loss = 0.0

        with torch.no_grad():
            for batch in tqdm(
                val_loader,
                desc=f"Epoch {epoch+1}/{args.epochs} [val]"
            ):
                graphs = batch["graph"]
                input_ids = batch["input_ids"].to(device)

                graph_batch = Batch.from_data_list(graphs).to(device)

                graph_emb = graph_encoder(graph_batch)
                soft_prompt = mapper(graph_emb)

                text_emb = llm.get_input_embeddings()(input_ids)

                B = input_ids.size(0)
                prompt_embeds_batch = prompt_embeds.repeat(B, 1, 1)

                inputs_embeds = torch.cat(
                    [
                        prompt_embeds_batch,
                        soft_prompt,
                        text_emb,
                    ],
                    dim=1,
                )


                labels = torch.cat(
                [
                    torch.full(
                        (B, prompt_embeds_batch.size(1) + soft_prompt.size(1)),
                        -100,
                        device=device,
                    ),
                    input_ids,
                ],
                dim=1,
            )



                outputs = llm(
                    inputs_embeds=inputs_embeds,
                    labels=labels,
                )

                val_loss += outputs.loss.item()

        val_loss /= len(val_loader)

        print(
    f"Epoch {epoch+1}/{args.epochs} | "
    f"train loss: {train_loss:.4f} | "
    f"val loss: {val_loss:.4f}"
)

        if val_loss < best_val_loss:
            print(
                f"New best model (val loss {best_val_loss:.4f} → {val_loss:.4f}), saving..."
            )
            best_val_loss = val_loss

            save_checkpoint(
                graph_encoder,
                mapper,
                path=args.checkpoint_path,
                epoch=epoch + 1,
                val_loss=val_loss,
            )



if __name__ == "__main__":
    main()


