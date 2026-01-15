# GPS-based Graph-to-Text Retrieval

This directory contains a graph-to-text retrieval pipeline based on the
**Graph Positioning System (GPS)** architecture.
The GPS encoder combines **local message passing** and **global attention**
to capture both fine-grained chemical structure and global molecular context.

This method extends the GINE baseline with a more expressive graph encoder,
while keeping the same contrastive training and retrieval setup.

---

## Overview

The pipeline consists of three main steps:

1. **Generate text embeddings** from molecular descriptions  
2. **Train a GPS graph encoder** using contrastive learning  
3. **Retrieve descriptions** for test molecules via nearest-neighbor search  

The full pipeline can be executed from the command line or via notebooks.

---

## Step 1 — Generate Text Embeddings

Text embeddings are generated from **full molecular descriptions** using a
Transformer-based sentence encoder.

```bash
python ../../retrieval/generate_description_embeddings.py \
  --data_dir ../../data/ \
  --out_dir data \
  --model sentence-transformers/all-MiniLM-L6-v2 \
  --batch_size 32 \
  --pooling mean

```

## Step 2 — Train the GPS Graph Encoder

The GPS encoder is trained to align molecular graph embeddings with text embeddings using a contrastive objective. 
The training setup remains identical to the GINE baseline, but uses
a more expressive graph architecture.

```bash
python ../../retrieval/gps_encoder/train.py \
  --data_dir ../../data/ \
  --train_emb ./data/train_embeddings_sentence-transformers_all-MiniLM-L6-v2.csv \
  --val_emb ./data/validation_embeddings_sentence-transformers_all-MiniLM-L6-v2.csv \
  --out_ckpt model.pt \
  --epochs 2
```

Output: ```model.pt```: trained GPS graph encoder checkpoint

## Step 3 — Retrieve Descriptions

After training, molecular descriptions are retrieved for test graphs
using nearest-neighbor search in the learned embedding space.

```bash
python ../../retrieval/simple_retrieval.py \
  --encoder gps \
  --model_path model.pt \
  --train_graphs ../../data/train_graphs.pkl \
  --test_graphs ../../data/test_graphs.pkl \
  --train_emb ./data/train_embeddings_sentence-transformers_all-MiniLM-L6-v2.csv \
  --output_csv ./submission.csv
```

Retrieval with top k can be performed by doing 

```bash
!python ../../retrieval/retrieval_top_k.py \
--encoder gps \
--model_path model.pt \
--train_graphs ../../data/train_graphs.pkl \
--test_graphs ../../data/test_graphs.pkl \
--train_emb ./data/train_embeddings_sentence-transformers_all-MiniLM-L6-v2.csv \
--output_csv ./submission.csv \
--topk 10
```