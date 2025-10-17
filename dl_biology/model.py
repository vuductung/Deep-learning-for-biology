import torch.nn as nn


class RNN(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_size,
        hidden_size,
        num_layers,
    ):
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.rnn = nn.RNN(embedding_size, hidden_size, num_layers, batch_first=False)
        self.h2o = nn.Linear(hidden_size, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        embedded = self.embedding(x)
        output, _ = self.rnn(embedded)
        final_hidden = output[-1, :, :]
        output = self.h2o(final_hidden)
        return self.relu(output)
