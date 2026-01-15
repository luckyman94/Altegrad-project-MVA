import torch
from torch.utils.data import Dataset

class GraphTextDataset(Dataset):
    def __init__(self, graph_dataset, tokenizer, max_length=128):
        self.graph_dataset = graph_dataset
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.graph_dataset)

    def __getitem__(self, idx):
        graph = self.graph_dataset[idx]  

    
        text = graph.description

        tokens = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        return {
            "graph": graph,
            "input_ids": tokens["input_ids"].squeeze(0),
            "attention_mask": tokens["attention_mask"].squeeze(0),
        }


class GraphOnlyDataset(torch.utils.data.Dataset):
    def __init__(self, graph_dataset):
        self.graphs = graph_dataset

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx]