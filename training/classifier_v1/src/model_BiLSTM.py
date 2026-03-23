import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


class BiLSTMClassifier(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim=128,
        hidden_dim=128,
        num_classes=5,
        num_layers=1,
        dropout=0.3,
    ):

        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=embedding_dim, padding_idx=0
        )
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, input_ids, lengths):
        """
        input_ids: [batch_size, seq_len]
        """

        embeddings = self.embedding(input_ids)

        packed_embeddings = pack_padded_sequence(
            embeddings, lengths=lengths.cpu(), batch_first=True, enforce_sorted=False
        )

        # lstm_out, _ = self.lstm(embeddings)
        _, (hidden, _) = self.lstm(packed_embeddings)

        forward_hidden = hidden[-2]
        backward_hidden = hidden[-1]

        combined_hidden = torch.cat((forward_hidden, backward_hidden), dim=1)

        output = self.dropout(combined_hidden)
        logits = self.fc(output)
        # last_hidden = lstm_out[:, -1, :]
        # output = self.dropout(last_hidden)
        # logits = self.fc(output)

        return logits
