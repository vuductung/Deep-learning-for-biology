from typing import Any

import h5py
import torch
from torch.utils.data import DataLoader, Dataset


def generate_dataloader(
    label_data,
    entry_labels,
    embedding_dir,
    train_seq_ids,
    test_seq_ids,
    collate_fn,
    batch_size,
    shuffle=True,
):
    train_dataset = Esm2EmbeddingDataset(label_data, entry_labels, train_seq_ids, embedding_dir)
    test_dataset = Esm2EmbeddingDataset(label_data, entry_labels, test_seq_ids, embedding_dir)

    return DataLoader[Any](train_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=shuffle), DataLoader(
        test_dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=False,
    )


class Esm2EmbeddingDataset(Dataset):
    def __init__(self, label_data, entry_labels, seq_ids, embedding_dir):
        super().__init__()
        self.embedding_dir = embedding_dir
        self.label_data = label_data
        self.seq_id_to_label_idx = {entry_id: idx for idx, entry_id in enumerate(entry_labels)}
        self.seq_ids = seq_ids
        self._h5_file = None

    def __len__(self):
        return len(self.seq_ids)

    def __getitem__(self, idx):
        seq_id = self.seq_ids[idx]

        if self._h5_file is None:
            self._h5_file = h5py.File(self.embedding_dir, "r", swmr=True)
        embedding = torch.from_numpy(self._h5_file[seq_id]["embedding"][()]).float()  # (seq_len, embedding_size)
        label_idx = self.seq_id_to_label_idx[seq_id]
        label = torch.as_tensor(self.label_data[label_idx], dtype=torch.float32)

        return {"seq_len": embedding.shape[0], "embedding": embedding, "label": label}
