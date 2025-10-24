import math
from turtle import position

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


class Rnn(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers, embedding_size=None):
        super(Rnn, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.rnn = nn.GRU(embedding_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.h2o = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.Linear(32, 16),
            nn.Linear(16, 1),
        )

    def forward(self, x, lengths):
        x = self.embedding(x)
        x_packed = pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )  # pad the sequence for batch training
        _, hidden = self.rnn(x_packed)  # (batch, sequence, 2*hidden), # (2*num_layer, hidden)
        final_hidden = hidden[-1]
        output = self.h2o(final_hidden)

        return output.squeeze(-1)  # (batch, 1) -> (batch,)


class TransformerEncoder(nn.Module):
    def __init__(self, d_model, dim_feedforward, nhead, dropout, max_len, activation="relu", batch_first=True):
        super(TransformerEncoder, self).__init__()

        self.d_model = d_model
        self.feedforward_dim = dim_feedforward
        self.nhead = nhead
        self.dropout = nn.Dropout(dropout)
        self.activation = activation
        self.batch_first = batch_first

        self.pe = PositionalEncoder(d_model, max_len, dropout)


class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_len, dropout):
        super(PositionalEncoder, self).__init__()

        self.d_model = d_model
        self.max_len = max_len
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)  # (max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (max_len//2, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))  # (max_len//2, )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        # x (batch, seq_len, d_model)
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)
