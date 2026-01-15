import torch
from torch.utils.data import Dataset
from pathlib import Path

import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from constants import LATENT_TOKENS, D_MODEL

class GraphLatentDataset(Dataset):
    def __init__(self, graphs, id2text, latent_ae):
        self.graphs = graphs
        self.texts = [id2text[g.id] for g in graphs]
        self.ae = latent_ae

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        g = self.graphs[idx]
        text = self.texts[idx]

        with torch.no_grad():
            latent = self.ae.encode([text])[0]

        if latent.size(0) < LATENT_TOKENS:
            pad = torch.zeros(LATENT_TOKENS - latent.size(0), D_MODEL)
            latent = torch.cat([latent, pad], 0)
        else:
            latent = latent[:LATENT_TOKENS]

        return g, latent, text
