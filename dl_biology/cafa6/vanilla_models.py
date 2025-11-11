import math

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


class NeuralNet(nn.Module):
    def __init__(self, embedding_size, dropout, out_dim):
        super(NeuralNet, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(embedding_size, embedding_size * 2),
            nn.BatchNorm1d(embedding_size * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_size * 2, embedding_size * 4),
            nn.BatchNorm1d(embedding_size * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_size * 4, embedding_size * 4),
            nn.BatchNorm1d(embedding_size * 4),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.output_head = nn.Sequential(
            nn.Linear(embedding_size * 4, embedding_size * 8),
            nn.BatchNorm1d(embedding_size * 8),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(embedding_size * 8, out_dim),
        )

    def forward(self, embedding, attn_mask):
        attn_mask = attn_mask.unsqueeze(-1)  # (batch_size, seq_len) -> (batch_size, seq_len, 1)
        masked_embedding = (embedding * attn_mask).sum(1)  # (batch_size, embedding_size)
        valid_counts = attn_mask.sum(1).clamp(min=1)  # (batch_size, 1), clamp to avoid division by zero
        embedding_mean = masked_embedding / valid_counts
        return self.output_head(self.encoder(embedding_mean))


class NeuralNetAttnPooling(nn.Module):
    def __init__(self, embedding_size, output_dim, dropout):
        super(NeuralNetAttnPooling, self).__init__()

        self.embedding_size = embedding_size
        self.attention = nn.Sequential(
            nn.Linear(embedding_size, embedding_size // 2),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(embedding_size // 2, 1),
        )

        self.linear = self.Sequential(
            nn.Linear(embedding_size, embedding_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_size // 2, output_dim),
        )

    # def forward(self, embedding, attn_mask):

    #     pooled_weights = self.attention(embedding) # (batch_size, seq_len, 1)
    #     embedding_agg = (embedding * pooled_weights).sum(dim=1) # (batch_size, embedding_size)
    #     return
