import math

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


class RnnRt(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers, embedding_size=None):
        super(RnnRt, self).__init__()
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
        output = self.h2o(final_hidden)  # (1, hidden) -> (batch, 1)

        return output.squeeze(-1)  # (batch, 1) -> (batch,)


class TransformerEncoderRt(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model,
        dim_feedforward,
        nhead,
        dropout,
        max_len,
        num_layers,
        activation="relu",
        batch_first=True,
    ):
        super(TransformerEncoderRt, self).__init__()

        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)

        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_encoder = PositionalEncoder(d_model, max_len, dropout)
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=batch_first,
        )
        self.transf_encoder = nn.TransformerEncoder(encoder_layer=self.encoder_layer, num_layers=num_layers)

        self.output_head = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

        self._init_weight()

    def _init_weight(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x, lengths):
        # x (batch, seq_len)
        seq_len = x.size(1)
        padding_mask = torch.arange(0, seq_len).reshape(1, -1) >= lengths.reshape(-1, 1)  # (batch, seq_len)
        x = self.embedding(x) * math.sqrt(self.d_model)  #  (batch, seq_len, d_model) (scale the data)
        x = self.pos_encoder(x)  # (batch, seq_len, d_model)
        x = self.transf_encoder(x, src_key_padding_mask=padding_mask)  # (batch, seq_len, d_model)

        mask = ~padding_mask.unsqueeze(-1)  # (batch, seq_len, 1)

        x = (x * mask).sum(dim=1) / lengths.unsqueeze(-1).float()  # (batch, d_model)
        x = self.output_head(x)  # x(batch, d_model) -> (batch, 1)
        return x.squeeze(-1)  # x(batch, 1) -> (batch, )


class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_len, dropout):
        super(PositionalEncoder, self).__init__()

        self.d_model = d_model
        self.max_len = max_len
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)  # (max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (max_len//2, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))  # (max_len//2, )

        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        # x (batch, seq_len, d_model)
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)
