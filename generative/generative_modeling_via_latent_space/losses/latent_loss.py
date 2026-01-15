import torch.nn.functional as F

def latent_loss(pred, target):
    mse = F.mse_loss(pred, target)
    cos = 1 - F.cosine_similarity(
        pred.flatten(1),
        target.flatten(1),
        dim=-1
    ).mean()
    return mse + 0.01 * cos
