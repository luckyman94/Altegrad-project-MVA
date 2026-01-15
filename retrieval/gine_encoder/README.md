# GINE-based Graph-to-Text Retrieval

This directory contains a **baseline retrieval pipeline** based on a
**Graph Isomorphism Network with Edge features (GINE)**.
The goal is to align molecular graph representations with text embeddings
and retrieve the most relevant molecular description via nearest-neighbor search.

This method serves as a **strong and simple baseline** for molecular graph captioning.

---

## Overview

The pipeline consists of three main steps:

1. **Generate text embeddings** from molecular descriptions  
2. **Train a GINE graph encoder** using contrastive learning  
3. **Retrieve descriptions** for test molecules via kNN search  

The full pipeline can be run either from notebooks or directly from the command line.

---

## Step 1 — Generate Text Embeddings

First, generate text embeddings from molecular descriptions using a
Transformer-based sentence encoder.

This step encodes **full molecular descriptions** into fixed-size vectors.

```bash
python ../../retrieval/generate_description_embeddings.py \
  --data_dir ../../data/ \
  --out_dir data \
  --model sentence-transformers/all-MiniLM-L6-v2 \
  --batch_size 32 \
  --pooling mean
```

This produces files such as:

```
data/
├── train_embeddings_sentence-transformers_all-MiniLM-L6-v2.csv
└── validation_embeddings_sentence-transformers_all-MiniLM-L6-v2.csv
```

## Step 2 — Train the GINE Graph Encoder

The graph encoder is trained to align molecular graph embeddings with
the corresponding text embeddings using a contrastive objective.

```bash
python ../../retrieval/gine_encoder/train.py \
  --train_graphs ../../data/train_graphs.pkl \
  --val_graphs ../../data/validation_graphs.pkl \
  --train_emb ./data/train_embeddings_sentence-transformers_all-MiniLM-L6-v2.csv \
  --val_emb ./data/validation_embeddings_sentence-transformers_all-MiniLM-L6-v2.csv \
  --output_model model.pt \
  --epochs 10
```

Output:
```model.pt```: trained GINE graph encoder checkpoint


## Step 3 — Retrieve Descriptions 

Once the graph encoder is trained, retrieve descriptions for test molecules via nearest-neighbor search in the joint embedding space.
You can choose your retrieval strategy between top k and simple retrieval by adapting the script and the args.

```bash
python ../../retrieval/simple_retrieval.py \
  --encoder gine \
  --model_path model.pt \
  --train_graphs ../../data/train_graphs.pkl \
  --test_graphs ../../data/test_graphs.pkl \
  --train_emb ./data/train_embeddings_sentence-transformers_all-MiniLM-L6-v2.csv \
  --output_csv ./submission.csv
```

