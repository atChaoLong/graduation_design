import torch
import torch.nn as nn
from transformers import AutoModel


class BERTForMultiLabel(nn.Module):
    """
    BERT baseline for multi-label classification.
    Simple BERT + Linear, no GCN.
    Input: token_ids (batch, seq_len), attention_mask (batch, seq_len)
    Output: logits (batch, num_labels)
    """

    def __init__(self, bert_model='bert-base-chinese', num_labels=100, dropout=0.1):
        super().__init__()
        self.bert = AutoModel.from_pretrained(bert_model)
        bert_hidden = self.bert.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(bert_hidden, num_labels)

    def forward(self, input_ids, attention_mask):
        bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        # Use CLS token
        if hasattr(bert_out, 'pooler_output') and bert_out.pooler_output is not None:
            cls = bert_out.pooler_output
        else:
            cls = bert_out.last_hidden_state[:, 0, :]
        x = self.dropout(cls)
        logits = self.fc(x)
        return logits
