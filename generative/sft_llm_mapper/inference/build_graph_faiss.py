import faiss
import torch
import numpy as np
from tqdm import tqdm

from data_utils import PreprocessedGraphDataset
from sft_llm_mapper.models.encoder import GraphEncoder, GraphEncoderConfig

@torch.no_grad()
def build_faiss_index(
    graph_encoder,
    train_ds,
    device,
    out_path,
):
    graph_encoder.eval().to(device)

    all_embs = []
    all_texts = []

    for g in tqdm(train_ds, desc="Encoding train graphs"):
        z = graph_encoder(g.to(device).unsqueeze(0))  # [1, D]
        z = z.cpu().numpy()
        z = z / np.linalg.norm(z, axis=1, keepdims=True)

        all_embs.append(z)
        all_texts.append(g.description)

    X = np.vstack(all_embs).astype("float32")

    dim = X.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(X)

    faiss.write_index(index, out_path)
    torch.save(all_texts, out_path + ".texts")

    print(f"✅ FAISS index saved to {out_path}")
