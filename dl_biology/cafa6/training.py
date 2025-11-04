from typing import Any

import h5py
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


def generate_dataloader(label_dir, embedding_dir, train_seq_ids, test_seq_ids, collate_fn, batch_size, shuffle):
    train_dataset = Esm2EmbeddingDataset(label_dir, embedding_dir, train_seq_ids)
    test_dataset = Esm2EmbeddingDataset(label_dir, embedding_dir, test_seq_ids)

    return DataLoader[Any](train_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=shuffle), DataLoader(
        test_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=shuffle
    )


class Esm2EmbeddingDataset(Dataset):
    def __init__(self, label_dir, embedding_dir, seq_ids):
        super().__init__()
        self.label_data = pd.read_csv(label_dir, delimiter="\t")
        self.embedding_dir = embedding_dir
        self.seq_ids = seq_ids

        # Truncate the label data to only those entries with embeddings
        self.label_data_trunc = self.label_data[self.label_data["EntryID"].isin(self.seq_ids)]
        # Compute multi-hot labels as a tensor
        self.label = torch.from_numpy(
            pd.crosstab(self.label_data_trunc["EntryID"], self.label_data_trunc["term"])
            .reindex(self.seq_ids, fill_value=0)
            .values
        ).float()
        # Store terms to recover class order, if needed
        self.terms = pd.unique(self.label_data_trunc["term"])

    def __len__(self):
        return len(self.seq_ids)

    def __getitem__(self, idx):
        seq_id = self.seq_ids[idx]

        with h5py.File(self.embedding_dir, "r") as h5f:
            embedding = torch.from_numpy(h5f[seq_id]["embedding"][()], dtype=torch.float32)

        label = self.label[idx]

        return {"seq_len": embedding.shape[0], "embedding": embedding, "label": label}
