import os
import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import LegalJSONLDataset, build_label_cooccurrence
from model import BertGCNForMultiLabel
from utils import normalize_adj


def compute_metrics(y_true, y_pred):
    from sklearn.metrics import f1_score
    micro = f1_score(y_true, y_pred, average='micro', zero_division=0)
    macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    if y_true.shape[0] == 0:
        acc = 0.0
    else:
        acc = (y_true == y_pred).all(axis=1).mean()
    return micro, macro, acc


def collate_fn(batch):
    import torch
    input_ids = torch.stack([b['input_ids'] for b in batch])
    attention_mask = torch.stack([b['attention_mask'] for b in batch])
    labels = torch.from_numpy(np.array([b['labels'] for b in batch])).to(torch.float32)
    raw_facts = [b.get('raw_fact', '') for b in batch]
    orig_accs = [b.get('orig_accusation', []) for b in batch]
    orig_rels = [b.get('orig_relevant_articles', []) for b in batch]
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels,
        'raw_facts': raw_facts,
        'orig_accs': orig_accs,
        'orig_rels': orig_rels,
    }


def evaluate(args):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger('evaluate')

    # load checkpoint
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f'Checkpoint not found: {ckpt_path}')
    ckpt = torch.load(str(ckpt_path), map_location='cpu')
    if 'label2idx' not in ckpt:
        raise KeyError('checkpoint does not contain label2idx')
    label2idx = ckpt['label2idx']
    num_labels = len(label2idx)
    logger.info(f'Number of labels (from checkpoint): {num_labels}')

    # build adjacency using training data co-occurrence if provided
    if args.train_path:
        cooc = build_label_cooccurrence(args.train_path, label2idx)
        adj = normalize_adj(cooc)
        adj = torch.tensor(adj, dtype=torch.float32)
    else:
        # fallback to identity adj
        adj = torch.eye(num_labels, dtype=torch.float32)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = BertGCNForMultiLabel(bert_model=args.bert_model, num_labels=num_labels, label_dim=args.label_dim)
    model.load_state_dict(ckpt['model_state'])
    model.to(device)
    adj = adj.to(device)

    # dataset and loader
    ds = LegalJSONLDataset(args.eval_path, label2idx, max_length=args.max_length)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    # create run folder
    start_time = datetime.now().strftime('%Y%m%d%H%M%S')
    logs_root = Path(args.output_dir) / 'logs'
    logs_root.mkdir(parents=True, exist_ok=True)
    run_dir = logs_root / start_time
    run_dir.mkdir(parents=True, exist_ok=True)

    # evaluate
    model.eval()
    y_true = []
    y_pred = []
    idx2label = {v: k for k, v in label2idx.items()}
    pred_file = run_dir / 'eval_predictions.txt'
    with open(pred_file, 'w', encoding='utf-8') as pf:
        with torch.no_grad():
            for batch in tqdm(loader, desc='Eval', leave=False):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].cpu().numpy()
                logits = model(input_ids=input_ids, attention_mask=attention_mask, adj=adj)
                probs = torch.sigmoid(logits).cpu().numpy()
                preds = (probs >= args.threshold).astype(int)
                y_true.append(labels)
                y_pred.append(preds)

                raw_facts = batch.get('raw_facts', [])
                orig_accs = batch.get('orig_accs', [])
                orig_rels = batch.get('orig_rels', [])
                for i in range(preds.shape[0]):
                    pred_idx = list(np.where(preds[i] == 1)[0])
                    pred_labels = [idx2label[idx] for idx in pred_idx]
                    pred_accs = [lbl.split('::', 1)[1] for lbl in pred_labels if lbl.startswith('accusation::')]
                    pred_articles = [lbl.split('::', 1)[1] for lbl in pred_labels if lbl.startswith('article::')]
                    fact_line = raw_facts[i] if i < len(raw_facts) else ''
                    orig_acc_line = ','.join(orig_accs[i]) if i < len(orig_accs) else ''
                    orig_rel_line = ','.join(orig_rels[i]) if i < len(orig_rels) else ''
                    pred_acc_line = ','.join(pred_accs)
                    pred_rel_line = ','.join(pred_articles)
                    pf.write(f"{fact_line}\n")
                    pf.write(f"原罪名: {orig_acc_line} 原法条: {orig_rel_line}\n")
                    pf.write(f"预测罪名: {pred_acc_line} 预测法条: {pred_rel_line}\n\n")

    y_true = np.vstack(y_true)
    y_pred = np.vstack(y_pred)
    micro, macro, acc = compute_metrics(y_true, y_pred)
    logger.info(f'Eval: Micro-F1={micro:.4f} Macro-F1={macro:.4f} Acc={acc:.4f}')

    metrics_path = run_dir / 'eval_metrics.txt'
    with open(metrics_path, 'w', encoding='utf-8') as mf:
        mf.write(f'micro_f1={micro:.6f}\n')
        mf.write(f'macro_f1={macro:.6f}\n')
        mf.write(f'accuracy={acc:.6f}\n')
        mf.write(f'checkpoint={ckpt_path}\n')
        mf.write(f'eval_path={args.eval_path}\n')
        mf.write(f'timestamp={start_time}\n')

    print('Evaluation complete.')
    print('Predictions:', pred_file)
    print('Metrics:', metrics_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate trained model on final test set')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model.pt')
    parser.add_argument('--train_path', type=str, default='dataset/train_strict20.json', help='path to train json to build adjacency (optional)')
    parser.add_argument('--eval_path', type=str, default='dataset/evalution_strict20.json')
    parser.add_argument('--output_dir', type=str, default='checkpoints')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--bert_model', type=str, default='bert-base-chinese')
    parser.add_argument('--label_dim', type=int, default=256)
    parser.add_argument('--max_length', type=int, default=256)
    args = parser.parse_args()
    evaluate(args)
