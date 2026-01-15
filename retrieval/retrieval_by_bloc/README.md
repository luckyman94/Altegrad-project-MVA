# Multi-Block Weighted Retrieval

This directory implements a **Multi-Block Weighted Retrieval** strategy for
molecular graph captioning.

Instead of retrieving descriptions using a single text embedding, this method
**decomposes molecular descriptions into semantic blocks** and performs
**block-wise kNN retrieval**, followed by a **weighted score fusion**.

Importantly, **the graph encoder is not retrained** for this method.
Multi-block retrieval is applied **only at inference time**.

---

## Overview

Each molecular description is split into four semantic blocks:

- `intro` — general definition
- `structure` — chemical / structural properties
- `role` — biological or functional role
- `taxonomy` — class membership or derivation

Each block is embedded independently using a text encoder.
At inference time, a **single graph embedding** is compared against
each block-specific text embedding space.
Block-wise top-k retrieval results are fused via a **weighted aggregation**.

---

## Pipeline

1. **Train a graph encoder** (GINE or GPS) using full descriptions  
2. **Generate block-specific text embeddings** 
3. **Perform top-k retrieval per block**  
4. **Fuse scores with block-specific weights**  
5. **Select the final description**

---

## Step 1 — Generate Block-wise Text Embeddings

Generate embeddings for each semantic.

```bash
python ../../retrieval/retrieval_by_bloc/generate_multiblock_embeddings.py \
  --model sentence-transformers/all-MiniLM-L6-v2 \
  --data_dir ../../data/ \
  --out_dir embeddings/ \
  --pooling mean \
  --batch_size 32
```


This produces:
```
embeddings/
├── train_intro_embeddings_*.csv
├── train_structure_embeddings_*.csv
├── train_role_embeddings_*.csv
└── train_taxonomy_embeddings_*.csv
```

## Step 2 — Train the encoder (GINE or GPS)

The encoder must be train on the whole embeddings description. The tokenizer used must be the same than the one for block embeddings.

### Step 2.1 - Generate the full embeddings 

```bash
python ../../retrieval/generate_description_embeddings.py \
  --data_dir ../../data/ \
  --out_dir embeddings/ \
  --model sentence-transformers/all-MiniLM-L6-v2 \
  --batch_size 32 \
  --pooling mean
```
### Step 2.2 - Train the encoder 

```bash
python ../../retrieval/gps_encoder/train.py \
--data_dir ../../data/ \
--train_emb ./embeddings/train_embeddings_sentence-transformers_all-MiniLM-L6-v2.csv \
--val_emb ./embeddings/validation_embeddings_sentence-transformers_all-MiniLM-L6-v2.csv \
--out_ckpt model.pt \
--epochs 10 
```

## Step 3 - Run multiblock pipeline retrieval

```bash
python ../../retrieval/retrieval_by_bloc/retrieval_multiblock.py \
  --encoder gps \
  --model_path model.pt \
  --train_graphs ../../data/train_graphs.pkl \
  --test_graphs ../../data/test_graphs.pkl \
  --train_intro_emb embeddings/train_intro_embeddings_sentence-transformers_all-MiniLM-L6-v2.csv \
  --train_structure_emb embeddings/train_structure_embeddings_sentence-transformers_all-MiniLM-L6-v2.csv \
  --train_role_emb embeddings/train_role_embeddings_sentence-transformers_all-MiniLM-L6-v2.csv \
  --train_taxonomy_emb embeddings/train_taxonomy_embeddings_sentence-transformers_all-MiniLM-L6-v2.csv \
  --output_csv submission.csv \
  --top_k 5

```
### Block Weights 

Each block contributes to the final score via a predefined weight:

```DEFAULT_BLOCK_WEIGHTS = {
    "intro": 1.0,
    "structure": 0.7,
    "role": 1.2,
    "taxonomy": 0.5,
}
```

Weights can be adjusted to emphasize specific semantic aspects.
