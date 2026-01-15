# Molecular Graph Captioning — Retrieval-based Approaches

This repository explores multiple **graph-to-text retrieval strategies** for molecular description generation.
The objective is to map molecular graphs to natural language descriptions by learning a shared semantic space
between graph representations and text embeddings.

The project investigates **three complementary methods**, ranging from simple graph encoders to more structured
semantic retrieval strategies.

---

## Methods Overview

### Method 1 — GINE-based Graph Encoder (Baseline Retrieval)

This method relies on a **Graph Isomorphism Network with Edge features (GINE)** to encode molecular graphs.
The graph embeddings are aligned with text embeddings using a contrastive (InfoNCE-style) objective.
At inference time, a nearest-neighbor retrieval is performed in the embedding space.

- Graph encoder: GINE
- Text encoder: Sentence Transformer / Transformer encoder
- Retrieval: mono-block kNN (full description)
- Role: strong and simple baseline

Directory: ```retrieval/gine_encoder```


---

### Method 2 - GPS-based Graph Encoder

A more expressive graph encoder based on the **Graph Positioning System (GPS)** architecture,
combining local message passing and global attention mechanisms.
This model captures both fine-grained chemical structure and global molecular context.

- Graph encoder: GPS (local + global attention)
- Text encoder: Transformer-based sentence encoder
- Retrieval: mono-block kNN (full description)

Directory: ```retrieval/gps_encoder```


---

### Method 3 - Multi-Block Weighted Retrieval

A structured retrieval strategy where molecular descriptions are decomposed into
semantic blocks:

- introduction
- structure
- role
- taxonomy

Each block is embedded independently. At inference time, a **single graph embedding**
is compared against each block-specific text embedding space.
Retrieval results are fused using a **weighted aggregation of similarity scores**.

This method does **not retrain the graph encoder** and operates purely at retrieval time.

- Graph encoder: GINE or GPS (unchanged)
- Text encoder: Transformer encoder (block-wise)
- Retrieval: block-wise top-k with weighted fusion

Directory: ```retrieval/retrieval_by_bloc```

