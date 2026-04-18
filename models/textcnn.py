import torch
import torch.nn as nn
import torch.nn.functional as F


class TextCNNForMultiLabel(nn.Module):
    """
    TextCNN for multi-label classification.
    Uses multi-scale convolution (filter sizes 3, 4, 5) to capture local features.
    Input: token_ids (batch, seq_len) from BERT tokenizer
    Output: logits (batch, num_labels)
    """

    def __init__(self, vocab_size, embed_dim=256, num_labels=100,
                 filter_sizes=[3, 4, 5], num_filters=256, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_labels = num_labels

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        # Multi-scale convolution
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, num_filters, fs, padding=fs // 2)
            for fs in filter_sizes
        ])

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Output layer
        self.fc = nn.Linear(len(filter_sizes) * num_filters, num_labels)

    def forward(self, input_ids):
        # input_ids: (batch, seq_len)
        batch_size = input_ids.size(0)

        # Embedding: (batch, seq_len, embed_dim)
        x = self.embedding(input_ids)

        # Conv1d expects (batch, channels, seq_len)
        x = x.transpose(1, 2)

        # Apply each conv and max pool
        conv_results = []
        for conv in self.convs:
            # conv: (batch, num_filters, seq_len')
            conv_out = conv(x)
            # ReLU
            conv_out = F.relu(conv_out)
            # Max pooling over sequence: (batch, num_filters)
            pooled = F.max_pool1d(conv_out, conv_out.size(2)).squeeze(2)
            conv_results.append(pooled)

        # Concatenate all: (batch, len(filter_sizes) * num_filters)
        x = torch.cat(conv_results, dim=1)

        x = self.dropout(x)
        logits = self.fc(x)
        return logits
