import argparse
import pickle
import os
from pathlib import Path
import re

import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, T5EncoderModel



def split_description(desc: str):
    blocks = {
        "intro": "",
        "structure": "",
        "role": "",
        "taxonomy": "",
    }

    desc = desc.strip()

    m = re.search(r"(The molecule is .*?\.)", desc)
    if m:
        blocks["intro"] = m.group(1)

    m = re.search(r"(It has a role as .*?\.)", desc)
    if m:
        blocks["role"] = m.group(1)

    m = re.search(
        r"(It is a .*?\."
        r"|It derives from .*?\."
        r"|It is a conjugate base of .*?\."
        r"|It is a tautomer of .*?\.)",
        desc
    )
    if m:
        blocks["taxonomy"] = m.group(1)

    used = " ".join(v for v in blocks.values() if v)
    remainder = desc.replace(used, "").strip()
    blocks["structure"] = remainder if remainder else desc

    for k in blocks:
        if not blocks[k]:
            blocks[k] = desc

    return blocks



def pool_embeddings(outputs, attention_mask, pooling):
    hidden = outputs.last_hidden_state

    if pooling == "cls":
        return hidden[:, 0]

    elif pooling == "mean":
        mask = attention_mask.unsqueeze(-1).float()
        summed = torch.sum(hidden * mask, dim=1)
        counts = torch.clamp(mask.sum(dim=1), min=1e-9)
        return summed / counts

    elif pooling == "last":
        lengths = attention_mask.sum(dim=1) - 1
        return hidden[torch.arange(hidden.size(0)), lengths]

    else:
        raise ValueError(f"Unknown pooling: {pooling}")


def sanitize_model_name(model_name: str) -> str:
    return model_name.replace("/", "_")



def main():
    parser = argparse.ArgumentParser(
        description="Generate text embeddings per description block"
    )

    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--pooling", type=str, choices=["cls", "mean", "last"], default="mean")
    parser.add_argument("--splits", nargs="+", default=["train", "validation"])

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if "t5" in args.model.lower():
        model = T5EncoderModel.from_pretrained(args.model)
    else:
        model = AutoModel.from_pretrained(args.model)

    model = model.to(device)
    model.eval()

    os.makedirs(args.out_dir, exist_ok=True)
    model_tag = sanitize_model_name(args.model)

    for split in args.splits:
        print(f"\n=== Processing split: {split} ===")

        pkl_path = Path(args.data_dir) / f"{split}_graphs.pkl"
        if not pkl_path.exists():
            print(f"⚠️ Missing {pkl_path}, skipping")
            continue

        with open(pkl_path, "rb") as f:
            graphs = pickle.load(f)

        texts_by_block = {
            "intro": [],
            "structure": [],
            "role": [],
            "taxonomy": [],
        }
        ids = []

        for g in graphs:
            blocks = split_description(g.description)
            for k in texts_by_block:
                texts_by_block[k].append(blocks[k])
            ids.append(g.id)

        for block_name, texts in texts_by_block.items():
            print(f"\nEncoding block: {block_name}")

            all_embeddings = []

            for i in tqdm(range(0, len(texts), args.batch_size)):
                batch_texts = texts[i : i + args.batch_size]

                if "e5" in args.model.lower():
                    batch_texts = [f"passage: {t}" for t in batch_texts]

                inputs = tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=args.max_length,
                    return_tensors="pt",
                ).to(device)

                with torch.no_grad():
                    outputs = model(**inputs)

                emb = pool_embeddings(
                    outputs,
                    inputs["attention_mask"],
                    args.pooling,
                )

                emb = torch.nn.functional.normalize(emb, dim=-1)
                all_embeddings.append(emb.cpu())

            all_embeddings = torch.cat(all_embeddings, dim=0)

            df = pd.DataFrame({
                "ID": ids,
                "embedding": [
                    ",".join(map(str, e.tolist()))
                    for e in all_embeddings
                ],
            })

            out_path = (
                Path(args.out_dir)
                / f"{split}_{block_name}_embeddings_{model_tag}.csv"
            )

            df.to_csv(out_path, index=False)
            print(f"Saved → {out_path}")

    print("All embeddings generated.")


if __name__ == "__main__":
    main()
