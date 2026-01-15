# Generative modeling via latent space

---

## Overview

This method proposes a **two-step graph-to-text generation pipeline** in which molecular graphs are first mapped to a **latent prefix representation**, and then decoded into natural language using a **pretrained and mostly frozen language model**.

Instead of training a graph-to-text model end-to-end, we **reuse the latent space of a pretrained T5 model** and learn a graph encoder that predicts the corresponding latent representations.  
This design stabilizes training and significantly reduces the number of trainable parameters.

---

## Core Idea

The pipeline relies on the following principles:

1. A pretrained **T5 encoder–decoder** defines a rich latent space for text.
2. A **Latent Prefix Autoencoder (AE)** learns to reconstruct text from a fixed number of encoder tokens.
3. A **Graph Neural Network (GINE-based)** is trained to predict these latent tokens directly from molecular graphs.
4. Text generation is performed by decoding the predicted latent tokens using the frozen T5 decoder.


---

## Architecture

### 1. Latent Prefix Autoencoder (Text → Latent → Text)

- Based on **T5-small**
- Text is encoded using the T5 encoder
- Only the first `LATENT_TOKENS` encoder states are kept
- A lightweight projection network (`from_enc`) is applied to define the latent prefix
- The decoder reconstructs the original text from this latent prefix

This model is pretrained once and then reused as a **fixed latent space**.

---

### 2. Graph → Latent Model

- Input: molecular graphs with node and edge features
- Architecture:
  - Linear node feature projection
  - 3 stacked **GINEConv** layers with residual connections and layer normalization
  - Global mean pooling
  - MLP readout producing `LATENT_TOKENS × D_MODEL` outputs
- Output: a latent prefix compatible with the T5 encoder hidden states

---

## 3. Training Objective

The graph-to-latent model is trained using a **hybrid loss**:

### Latent alignment loss
- Mean Squared Error (MSE) between predicted and target latent prefixes
- Cosine similarity regularization to preserve global structure

### Decoder reconstruction loss
- Cross-entropy loss from the T5 decoder
- The predicted latent prefix is used as encoder input
- Only the last decoder layers may be unfrozen

The directory that correspond to this method is 
```generative_modeling_via_latent_space```

**WARNING** : Make sure to put the correct DATA_DIR path in the load_data file




