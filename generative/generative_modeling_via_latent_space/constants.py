import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LATENT_TOKENS = 192
D_MODEL = 512
TRAIN_MAX_LEN = 384
GEN_MAX_LEN = 640
MODEL_NAME = "t5-small"
BATCH_SIZE = 8
SAVE_DIR = "./graph2latent_ckpts"
