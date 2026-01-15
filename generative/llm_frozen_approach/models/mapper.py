import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearMapper(nn.Module):
    def __init__(self, dim_graph, dim_llm, num_soft_tokens):
        super().__init__()
        self.dim_graph = dim_graph
        self.dim_llm = dim_llm
        self.num_soft_tokens = num_soft_tokens

        self.linear = nn.Linear(
            dim_graph,
            dim_llm * num_soft_tokens
        )

    def forward(self, graph_emb):
        B = graph_emb.size(0)
        out = self.linear(graph_emb)
        return out.view(B, self.num_soft_tokens, self.dim_llm)