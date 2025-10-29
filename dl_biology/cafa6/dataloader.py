import torch
from torch.utils.data import Dataset


class PeptideDataset(Dataset):
    def __init__(self, pep_sequence, retention_time):
        self.pep_seq = pep_sequence
        self.rt = torch.tensor(retention_time, dtype=torch.float32)

    def __len__(self):
        return len(self.pep_seq)

    def __getitem__(self, idx):
        return {"pep_seq": self.pep_seq[idx], "rt": self.rt[idx]}
