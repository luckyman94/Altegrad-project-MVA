import numpy as np
from tqdm import tqdm
from nltk.translate.bleu_score import corpus_bleu
from rouge_score import rouge_scorer
import torch

from constants import DEVICE

@torch.no_grad()
def evaluate_bleu_rouge(model, loader, latent_ae, max_print=10):
    model.eval()
    refs, hyps = [], []

    for graph, latent, _ in tqdm(loader):
        graph = graph.to(DEVICE)

        preds = latent_ae.decode(model(graph))
        golds = latent_ae.decode(latent.to(DEVICE))

        for r, h in zip(golds, preds):
            refs.append([r.split()])
            hyps.append(h.split())

    bleu = corpus_bleu(refs, hyps)

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    rouge = np.mean([
        scorer.score(" ".join(r[0]), " ".join(h))["rougeL"].fmeasure
        for r, h in zip(refs, hyps)
    ])

    return bleu, rouge
