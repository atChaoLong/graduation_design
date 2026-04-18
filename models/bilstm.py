import torch
import torch.nn as nn
import torch.nn.functional as F


class BiLSTMForMultiLabel(nn.Module):
    """
    BiLSTM for multi-label classification.
    Bidirectional LSTM to capture sequential context.
    Input: token_ids (batch, seq_len) from BERT tokenizer
    Output: logits (batch, num_labels)
    """

    def __init__(self, vocab_size, embed_dim=256, hidden_dim=256,
                 num_layers=2, num_labels=100, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        # BiLSTM
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Output layer: hidden_dim * 2 (bidirectional)
        self.fc = nn.Linear(hidden_dim * 2, num_labels)

    def forward(self, input_ids):
        # input_ids: (batch, seq_len)
        batch_size = input_ids.size(0)

        # Embedding: (batch, seq_len, embed_dim)
        x = self.embedding(input_ids)

        # LSTM: output (batch, seq_len, hidden_dim*2)
        lstm_out, (hidden, cell) = self.lstm(x)

        # Use the last hidden state of both directions
        # hidden: (num_layers*2, batch, hidden_dim)
        # Concatenate forward and backward last layers
        forward_hidden = hidden[-2, :, :]  # (batch, hidden_dim)
        backward_hidden = hidden[-1, :, :]  # (batch, hidden_dim)
        context = torch.cat([forward_hidden, backward_hidden], dim=1)  # (batch, hidden_dim*2)

        x = self.dropout(context)
        logits = self.fc(x)
        return logits
