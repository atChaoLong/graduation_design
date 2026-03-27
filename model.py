import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x, adj):
        # x: (num_labels, in_dim), adj: (num_labels, num_labels)
        x = torch.matmul(adj, x)
        return F.relu(self.linear(x))


class BertGCNForMultiLabel(nn.Module):
    def __init__(self, bert_model='bert-base-chinese', num_labels=100, label_dim=256):
        super().__init__()
        self.bert = AutoModel.from_pretrained(bert_model)
        bert_hidden = self.bert.config.hidden_size
        self.proj = nn.Linear(bert_hidden, label_dim)
        self.label_emb = nn.Parameter(torch.randn(num_labels, label_dim))
        self.gcn1 = GCNLayer(label_dim, label_dim)
        self.gcn2 = GCNLayer(label_dim, label_dim)

    def forward(self, input_ids, attention_mask, adj):
        bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        if hasattr(bert_out, 'pooler_output') and bert_out.pooler_output is not None:
            cls = bert_out.pooler_output
        else:
            cls = bert_out.last_hidden_state[:, 0, :]
        doc = self.proj(cls)  # (batch, label_dim)

        x = self.label_emb  # (num_labels, label_dim)
        x = self.gcn1(x, adj)
        x = self.gcn2(x, adj)

        logits = torch.matmul(doc, x.t())
        return logits
