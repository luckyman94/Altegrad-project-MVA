import torch
import torch.nn.functional as F

def infonce_loss(mol_vec, txt_vec, temperature=0.07):
    logits = (mol_vec @ txt_vec.t()) / temperature

    batch_size = mol_vec.size(0)
    device = mol_vec.device
    labels = torch.arange(batch_size, device=device)

    loss_i = F.cross_entropy(logits, labels)
    loss_t = F.cross_entropy(logits.t(), labels)

    return 0.5 * (loss_i + loss_t)
