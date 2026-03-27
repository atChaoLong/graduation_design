import json
import numpy as np
import torch

def load_jsonl(path):
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def normalize_adj(adj):
    """Symmetric normalization: D^-0.5 (A+I) D^-0.5"""
    A = adj + np.eye(adj.shape[0], dtype=adj.dtype)
    deg = A.sum(axis=1)
    deg_inv_sqrt = np.power(deg, -0.5)
    deg_inv_sqrt[np.isinf(deg_inv_sqrt)] = 0.0
    D_inv_sqrt = np.diag(deg_inv_sqrt)
    return D_inv_sqrt @ A @ D_inv_sqrt

def save_checkpoint(state, path):
    torch.save(state, path)
