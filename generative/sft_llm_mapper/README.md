# Molecular Graph Captioning — Generative Pipeline (Soft Prompts, LoRA & Graph-RAG)

This repository implements a **graph-to-text generative pipeline** for molecular description generation.
The system maps molecular graphs to natural language descriptions using a **frozen graph encoder**,
a **soft-prompt mapper**, a **large language model fine-tuned with LoRA** and an optional
**graph-based Retrieval-Augmented Generation (RAG)** module.


---


## Training Pipeline Overview

The full pipeline is decomposed into **three training stages**.

### Stage 0 — Graph Encoder Training (Contrastive)

A graph encoder is trained to align molecular graphs with text embeddings
using a **contrastive InfoNCE objective**.

- Encoder: GPS / GNN-based
- Objective: graph ↔ text alignment
- Output: frozen graph encoder

---

### Stage 1 — Graph → LLM Alignment (Soft Prompt Mapper)

A mapper is trained to project graph embeddings into the **LLM embedding space**.

- Graph encoder: frozen
- LLM: frozen (used only as an embedder)
- Mapper: trainable
- Loss: InfoNCE


---

### Stage 2 — Supervised Fine-Tuning (SFT + LoRA)

The language model is trained to generate full molecular descriptions.

- Graph encoder: frozen
- Mapper: trainable
- LLM: fine-tuned **only through LoRA adapters**
- Loss: causal language modeling (cross-entropy)


---

## Inference Modes

The pipeline supports **two inference modes**.

---

## Inference Without RAG (Pure Generation)

In this mode, the model generates descriptions **only from the graph**.

Pipeline:
1. Graph → graph encoder
2. Graph embedding → soft prompt mapper
3. Soft tokens injected as prefix
4. LLM generates the description

The notebook ```example_notebook.ipynb``` explains the whole pipeline

## Graph-Based RAG (Retrieval-Augmented Generation)

### Key Idea

Instead of retrieving text using keywords, **retrieval is performed in graph space**.

At inference time:
- Test graph embeddings are compared to **training graph embeddings**
- The closest training graphs are retrieved
- Their **descriptions are injected as additional context** into the LLM

The notebook ```example_notebook_rag.ipynb``` explains the whole pipeline

**WARNING** : All the paths have to be adjusted. In the notebook they correspond to my paths in Google Colab

The directory that contains all the code is ```sft_llm_mapper```






