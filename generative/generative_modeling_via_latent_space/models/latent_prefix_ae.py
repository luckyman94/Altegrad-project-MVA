import torch
import torch.nn as nn
from pathlib import Path

import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))
from transformers import T5ForConditionalGeneration, T5Tokenizer
from transformers.modeling_outputs import BaseModelOutput

from generative_modeling_via_latent_space.constants import (
    DEVICE, MODEL_NAME,
    TRAIN_MAX_LEN, GEN_MAX_LEN,
    LATENT_TOKENS, D_MODEL
)

class LatentPrefixAE(nn.Module):
    def __init__(self):
        super().__init__()

        self.tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
        self.model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)

        # MUST EXIST (checkpoint depends on it)
        self.from_enc = nn.Sequential(
            nn.LayerNorm(D_MODEL),
            nn.Linear(D_MODEL, D_MODEL),
            nn.GELU(),
            nn.Linear(D_MODEL, D_MODEL),
        )

        self.to(DEVICE)
        self.model.to(DEVICE)

    @torch.no_grad()
    def encode(self, texts):
        tok = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=TRAIN_MAX_LEN,
            return_tensors="pt"
        ).to(DEVICE)

        enc = self.model.encoder(
            input_ids=tok.input_ids,
            attention_mask=tok.attention_mask
        ).last_hidden_state

        return self.from_enc(enc[:, :LATENT_TOKENS])

    @torch.no_grad()
    def decode(self, latent):
        out = self.model.generate(
            encoder_outputs=BaseModelOutput(last_hidden_state=latent),
            max_length=GEN_MAX_LEN,
            num_beams=1,
            do_sample=False
        )
        return self.tokenizer.batch_decode(out, skip_special_tokens=True)