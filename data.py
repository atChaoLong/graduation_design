import os
from torch.utils.data import Dataset
import numpy as np
from transformers import AutoTokenizer
from utils import load_jsonl


class LegalJSONLDataset(Dataset):
    def __init__(self, path, label2idx, tokenizer_name='bert-base-chinese', max_length=256):
        self.samples = list(load_jsonl(path))
        self.label2idx = label2idx
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        fact = s.get('fact','')
        inputs = self.tokenizer(fact, truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt')
        labels = np.zeros(len(self.label2idx), dtype=np.float32)
        # 修复后的标签填充逻辑
        # support multiple data formats: top-level or inside meta
        acc = s.get('accusation', [])
        if (not acc) and isinstance(s.get('meta'), dict):
            acc = s['meta'].get('accusation', [])
        if isinstance(acc, str):
            acc = [acc]
        for a in acc:
            key = f'accusation::{a}' # 必须加前缀
            if key in self.label2idx:
                labels[self.label2idx[key]] = 1.0
        rel = s.get('relevant_articles', [])
        if (not rel) and isinstance(s.get('meta'), dict):
            rel = s['meta'].get('relevant_articles', [])
        if isinstance(rel, (str, int)):
            rel = [str(rel)]
        for r in rel:
            key = f'article::{str(r)}' # 必须加前缀
            if key in self.label2idx:
                labels[self.label2idx[key]] = 1.0

        # use normalized acc/rel (which may come from meta) for downstream display
        orig_acc = [str(x) for x in (acc or [])]
        orig_rel = [str(x) for x in (rel or [])]

        item = {
            'input_ids': inputs['input_ids'].squeeze(0),
            'attention_mask': inputs['attention_mask'].squeeze(0),
            'labels': labels,
            'raw_fact': fact,
            'orig_accusation': orig_acc,
            'orig_relevant_articles': orig_rel,
        }
        return item


def build_label_space(train_path):
    # collect unique labels from accusation and relevant_articles
    labels = []
    for s in load_jsonl(train_path):
        acc = s.get('accusation', [])
        if (not acc) and isinstance(s.get('meta'), dict):
            acc = s['meta'].get('accusation', [])
        if isinstance(acc, str):
            acc = [acc]
        for a in acc:
            labels.append(f'accusation::{a}')
        rel = s.get('relevant_articles', [])
        if (not rel) and isinstance(s.get('meta'), dict):
            rel = s['meta'].get('relevant_articles', [])
        if isinstance(rel, (str,int)):
            rel = [str(rel)]
        for r in rel:
            labels.append(f'article::{r}')
    unique = sorted(set(labels))
    label2idx = {lab: i for i, lab in enumerate(unique)}
    return label2idx


def build_label_cooccurrence(train_path, label2idx):
    n = len(label2idx)
    mat = np.zeros((n, n), dtype=np.float32)
    for s in load_jsonl(train_path):
        labs = []
        acc = s.get('accusation', [])
        if (not acc) and isinstance(s.get('meta'), dict):
            acc = s['meta'].get('accusation', [])
        if isinstance(acc, str):
            acc = [acc]
        for a in acc:
            key = f'accusation::{a}'
            if key in label2idx:
                labs.append(label2idx[key])
        rel = s.get('relevant_articles', [])
        if (not rel) and isinstance(s.get('meta'), dict):
            rel = s['meta'].get('relevant_articles', [])
        if isinstance(rel, (str,int)):
            rel = [str(rel)]
        for r in rel:
            key = f'article::{r}'
            if key in label2idx:
                labs.append(label2idx[key])
        labs = list(set(labs))
        for i in labs:
            for j in labs:
                mat[i, j] += 1.0
    return mat
