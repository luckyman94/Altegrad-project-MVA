# LLM-Frozen Graph-to-Text Generation (Soft Prompting)

This approach aims to generate textual molecular descriptions by **projecting molecular graphs into the latent space of a pre-trained Large Language Model (LLM)**, while keeping the LLM **entirely frozen**.

Instead of fine-tuning the language model, we learn a **graph-conditioned soft prompt** that allows the frozen LLM to produce text conditioned on molecular structure.

---

## Overview

The pipeline is composed of three main components:

- A **Graph Encoder** (GPS / GINE-based) that maps molecular graphs to fixed-size embeddings
- A **Mapper** that projects graph embeddings into a set of *soft prompt tokens*
- A **frozen LLM** (e.g. GPT-2, BioT5, BioGPT) that generates text conditioned on these soft prompts

Throughout the entire training process, **the LLM parameters remain fixed**.

---



The Mapper outputs a small number of learnable vectors (*soft tokens*) that are prepended to the LLM input embeddings and act as a continuous prompt encoding molecular structure.

---

## Training Setup

Two training configurations are supported:

### 1. LLm that are decoder only (BioGPT, GPT)

The whole pipeline is explained in ```example_notebook.ipynb```

---

### 2. LLM that are encoder decoder (BioT5)

The whole pipeline is explained in ```example_notebook_biot5.ipynb```


**WARNING** : All the paths have to be adjusted. In the notebook they correspond to my paths in Google Colab
---

## Learning Objective

Training is performed using **cross-entropy loss** on the generated text. The **cross-entropy-loss** is performed between the ground truth text and the generated text.

The directory that contains all the code is ```llm_frozen_approach```



