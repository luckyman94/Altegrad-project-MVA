# ALTEGRAD Project — MVA (ENS Paris-Saclay)

This repository contains the project developed for the **ALTEGRAD** course (*Advanced Learning for Text and Graph Data*) of the **Master MVA** (**Mathématiques, Vision, Apprentissage**) at **ENS Paris-Saclay**.

The objective of the project is to explore different approaches for **molecular graph-to-text generation**, by leveraging both **retrieval-based** and **generative** paradigms.

---

## Project Overview

Given a molecular graph as input, the task consists in producing a **natural language description** of the molecule.  
We investigate two fundamentally different strategies:

1. **Retrieval-based approaches**  
2. **Generative approaches based on latent alignment with language models**

Each approach is implemented independently, allowing a clear comparison of modeling choices, inductive biases and performance trade-offs.

---

## Repository Structure

The project is organized into two main components:

```text
Altegrad-project-MVA/
├── retrieval/      # Retrieval-based graph-to-text methods
├── generative/     # Generative graph-to-text methods
├── data/           # Datasets and preprocessing utilities
└── README.md
```

In each directory there is an other README to help you to understand what we have implemented
